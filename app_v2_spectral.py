import torch
import torchaudio
import numpy as np
from flask import Flask, render_template, request, jsonify
import io
import os
import logging
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import pickle
import umap
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)

class LatentSpaceExplorer:
    def __init__(self, n_points=64):  # Increased points for better visualization
        self.n_points = n_points
        self.save_dir = 'static'
        self.points_file = os.path.join(self.save_dir, 'audio_points.pkl')
        self.mapping_file = os.path.join(self.save_dir, 'latent_mapping.json')
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.device = torch.device("cpu")
        self.mps_device = torch.device("mps")
        
        logger.info("Loading model...")
        self.model = MusicgenForConditionalGeneration.from_pretrained(
            "facebook/musicgen-small",
            torch_dtype=torch.float32
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        
        # Load or generate points and their mappings
        self.audio_points = self.load_or_generate_points()
        self.latent_mapping = self.create_latent_mapping()
        
    def create_latent_mapping(self):
        """Create 2D mapping of latent points using both UMAP and t-SNE"""
        try:
            if os.path.exists(self.mapping_file):
                with open(self.mapping_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load mapping: {e}")
        
        # Convert audio tensors to feature vectors
        features = []
        for point in self.audio_points.values():
            # Ensure point is on CPU
            point = point.cpu()
            
            # Calculate basic statistics
            features_vec = [
                float(point.mean()),
                float(point.std()),
                float(point.max()),
                float(point.min()),
                float(torch.median(point)),
                # Add more sophisticated features
                float((point > 0).float().mean()),  # Zero crossing rate approx
                float(torch.abs(point).mean()),     # Average amplitude
            ]
            
            # Add frequency domain features if possible
            if point.shape[-1] >= 512:  # Only if we have enough samples
                # Take FFT of chunks and average
                chunk_size = 512
                stride = chunk_size // 2
                chunks = point.unfold(-1, chunk_size, stride)
                
                # Compute FFT for each chunk
                ffts = torch.fft.rfft(chunks, dim=-1)
                magnitudes = torch.abs(ffts)
                
                # Average frequency domain features
                avg_magnitudes = magnitudes.mean(0)
                
                # Add some frequency domain features
                n_bands = 8
                band_size = avg_magnitudes.shape[-1] // n_bands
                for i in range(n_bands):
                    start = i * band_size
                    end = start + band_size
                    features_vec.append(float(avg_magnitudes[start:end].mean()))
            
            features.append(features_vec)
        
        features = np.array(features)
        
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Generate UMAP embedding
        umap_reducer = umap.UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=min(15, len(features) - 1),  # Adjust for small datasets
            min_dist=0.1
        )
        umap_embedding = umap_reducer.fit_transform(features_scaled)
        
        # Generate t-SNE embedding
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=min(15, len(features) - 1),  # Adjust for small datasets
        )
        tsne_embedding = tsne.fit_transform(features_scaled)
        
        # Normalize embeddings to [0, 1] range
        def normalize_embedding(embedding):
            min_vals = embedding.min(axis=0)
            max_vals = embedding.max(axis=0)
            return (embedding - min_vals) / (max_vals - min_vals)
        
        umap_normalized = normalize_embedding(umap_embedding)
        tsne_normalized = normalize_embedding(tsne_embedding)
        
        # Create mapping dictionary
        mapping = {
            'umap': {
                f'point_{i}': {
                    'x': float(umap_normalized[i][0]),
                    'y': float(umap_normalized[i][1])
                } for i in range(len(umap_normalized))
            },
            'tsne': {
                f'point_{i}': {
                    'x': float(tsne_normalized[i][0]),
                    'y': float(tsne_normalized[i][1])
                } for i in range(len(tsne_normalized))
            }
        }
        
        # Save mapping
        with open(self.mapping_file, 'w') as f:
            json.dump(mapping, f)
        
        return mapping

    def spectral_interpolate(self, audio1, audio2, ratio):
        """Interpolate between two audio signals in the frequency domain"""
        logger.info(f"Performing spectral interpolation with ratio {ratio}")
        
        # Parameters for STFT
        n_fft = 2048
        hop_length = 512
        win_length = n_fft
        window = torch.hann_window(win_length).to(audio1.device)

        # Compute STFTs
        stft1 = torch.stft(audio1, 
                          n_fft=n_fft, 
                          hop_length=hop_length,
                          win_length=win_length,
                          window=window,
                          return_complex=True)
        
        stft2 = torch.stft(audio2, 
                          n_fft=n_fft, 
                          hop_length=hop_length,
                          win_length=win_length,
                          window=window,
                          return_complex=True)

        # Interpolate magnitudes logarithmically
        magnitudes1 = torch.abs(stft1)
        magnitudes2 = torch.abs(stft2)
        
        # Log-scale interpolation for better perceptual results
        log_magnitudes1 = torch.log(magnitudes1 + 1e-8)
        log_magnitudes2 = torch.log(magnitudes2 + 1e-8)
        interp_magnitudes = torch.exp((1 - ratio) * log_magnitudes1 + ratio * log_magnitudes2)

        # Phase interpolation
        phases1 = torch.angle(stft1)
        phases2 = torch.angle(stft2)
        
        # Unwrap phases for smoother interpolation
        phases_diff = (phases2 - phases1 + np.pi) % (2 * np.pi) - np.pi
        interp_phases = phases1 + ratio * phases_diff

        # Combine magnitude and phase
        interp_stft = interp_magnitudes * torch.exp(1j * interp_phases)

        # Convert back to time domain
        audio = torch.istft(interp_stft,
                          n_fft=n_fft,
                          hop_length=hop_length,
                          win_length=win_length,
                          window=window)

        return audio

    def interpolate_position(self, x, y, method='umap'):
        """Interpolate audio at any 2D position using spectral interpolation"""
        try:
            # Get all points and their positions
            positions = self.latent_mapping[method]
            
            # Calculate distances to all points
            distances = {}
            for point_id, pos in positions.items():
                dist = np.sqrt((x - pos['x'])**2 + (y - pos['y'])**2)
                distances[point_id] = dist
            
            # Get nearest points and weights
            nearest_points = sorted(distances.items(), key=lambda x: x[1])[:4]
            logger.info(f"Nearest points: {[p[0] for p in nearest_points]}")
            
            # Calculate weights using inverse distance weighting
            total_weight = sum(1/d for _, d in nearest_points)
            weights = [1/d/total_weight for _, d in nearest_points]
            logger.info(f"Interpolation weights: {weights}")

            # Get first point as reference
            ref_point = self.audio_points[nearest_points[0][0]].to(self.mps_device)
            if len(ref_point.shape) == 1:
                ref_point = ref_point.unsqueeze(0)

            # Progressive interpolation through all points
            result = ref_point
            accumulated_weight = weights[0]

            for i in range(1, len(nearest_points)):
                if accumulated_weight >= 1.0:
                    break

                next_point = self.audio_points[nearest_points[i][0]].to(self.mps_device)
                if len(next_point.shape) == 1:
                    next_point = next_point.unsqueeze(0)

                # Calculate relative weight for this interpolation
                relative_weight = weights[i] / (accumulated_weight + weights[i])
                
                # Match lengths
                min_length = min(result.size(-1), next_point.size(-1))
                result = result[..., :min_length]
                next_point = next_point[..., :min_length]

                # Spectral interpolation
                result = self.spectral_interpolate(result, next_point, relative_weight)
                accumulated_weight += weights[i]

            # Normalize output
            result = result / torch.max(torch.abs(result))
            
            logger.info(f"Generated interpolated audio with shape {result.shape}")
            return result.cpu()

        except Exception as e:
            logger.error(f"Interpolation error: {e}")
            raise

    def save_audio(self, tensor, filepath):
        """Save audio tensor with proper shape handling"""
        try:
            # Ensure tensor is on CPU
            tensor = tensor.cpu()
            
            # Normalize
            tensor = tensor / torch.max(torch.abs(tensor))
            
            # Ensure correct shape [channels, samples]
            if len(tensor.shape) == 1:
                tensor = tensor.unsqueeze(0)
            elif len(tensor.shape) > 2:
                tensor = tensor.squeeze()
                if len(tensor.shape) == 1:
                    tensor = tensor.unsqueeze(0)
            
            # Save audio
            torchaudio.save(
                filepath if isinstance(filepath, str) else filepath.name,
                tensor,
                self.model.config.audio_encoder.sampling_rate
            )
            return True
        except Exception as e:
            logger.error(f"Error saving audio: {e}")
            return False

    def generate_initial_points(self):
        """Generate initial audio points"""
        points = {}
        # Extended set of descriptions for more variety
        descriptions = [
            "ambient texture soft", "drum pattern tight", "melodic synth bright",
            "deep bass warm", "bell tones crystal", "noise texture rough",
            "rhythm glitch fast", "water flowing gentle", "space drone wide",
            "glitch beats sharp", "chord pad smooth", "high frequency ring",
            "bass pulse deep", "abstract sound chaos", "minimal beat clean",
            "ethereal voice float"
        ] * 4  # Repeat for more points
        
        with torch.no_grad():
            for i, desc in enumerate(descriptions[:self.n_points]):
                logger.info(f"Generating point {i}: {desc}")
                inputs = self.processor(
                    text=[desc],
                    padding=True,
                    return_tensors="pt"
                )
                
                audio = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    guidance_scale=3.0
                )
                
                filepath = os.path.join(self.save_dir, f"point_{i}.wav")
                self.save_audio(audio[0], filepath)
                points[f"point_{i}"] = audio[0]
        
        return points
    
    def load_or_generate_points(self):
        """Load existing points or generate new ones"""
        try:
            if os.path.exists(self.points_file):
                logger.info("Loading existing points...")
                with open(self.points_file, 'rb') as f:
                    points = pickle.load(f)
                if self.verify_points(points):
                    logger.info("Successfully loaded existing points")
                    return points
                else:
                    logger.warning("Loaded points verification failed")
            else:
                logger.info("No existing points found")
        except Exception as e:
            logger.error(f"Error loading points: {e}")

        # Generate new points if loading failed or file doesn't exist
        logger.info("Generating new points...")
        points = self.generate_initial_points()
        
        # Save the new points
        try:
            with open(self.points_file, 'wb') as f:
                pickle.dump(points, f)
            logger.info("Successfully saved new points")
        except Exception as e:
            logger.error(f"Error saving points: {e}")
            
        return points

    def verify_points(self, points):
        """Verify that loaded points are valid"""
        try:
            if not isinstance(points, dict):
                return False
            if len(points) == 0:
                return False
            # Verify first point is a tensor
            first_point = next(iter(points.values()))
            if not isinstance(first_point, torch.Tensor):
                return False
            return True
        except:
            return False


    def interpolate_audio(self, point1_idx, point2_idx, ratio):
        """Interpolate between two points"""
        point1 = self.audio_points[f"point_{point1_idx}"].to(self.mps_device)
        point2 = self.audio_points[f"point_{point2_idx}"].to(self.mps_device)
        
        min_length = min(point1.size(-1), point2.size(-1))
        point1 = point1[..., :min_length]
        point2 = point2[..., :min_length]
        
        interpolated = (1 - ratio) * point1 + ratio * point2
        interpolated = interpolated.cpu()
        
        # Normalize and prepare for saving
        interpolated = interpolated / torch.max(torch.abs(interpolated))
        if len(interpolated.shape) == 1:
            interpolated = interpolated.unsqueeze(0)
        
        buffer = io.BytesIO()
        torchaudio.save(
            buffer,
            interpolated,
            self.model.config.audio_encoder.sampling_rate,
            format='wav'
        )
        buffer.seek(0)
        return buffer

@app.route('/')
def index():
    return render_template('latent_explorer.html', n_points=explorer.n_points)

@app.route('/interpolate', methods=['POST'])
def interpolate():
    data = request.json
    try:
        buffer = explorer.interpolate_audio(
            str(data['point1']), 
            str(data['point2']), 
            float(data['ratio'])
        )
        
        filepath = 'static/interpolated.wav'
        with open(filepath, 'wb') as f:
            f.write(buffer.getvalue())
            
        return jsonify({
            'audio_url': f'/static/interpolated.wav?t={np.random.randint(10000)}'
        })
    except Exception as e:
        logger.error(f"Interpolation error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/check_audio/<filename>')
def check_audio(filename):
    filepath = f'static/{filename}'
    exists = os.path.exists(filepath)
    return jsonify({
        'exists': exists,
        'size': os.path.getsize(filepath) if exists else 0
    })
    
    
@app.route('/map')
def get_latent_map():
    return jsonify(explorer.latent_mapping)

@app.route('/generate_at_position', methods=['POST'])
def generate_at_position():
    data = request.json
    x = float(data['x'])
    y = float(data['y'])
    method = data.get('method', 'umap')
    
    try:
        audio = explorer.interpolate_position(x, y, method)
        
        # Save interpolated audio
        filepath = 'static/interpolated.wav'
        buffer = io.BytesIO()
        
        # Ensure audio has correct shape before saving
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)
        
        torchaudio.save(
            buffer,
            audio,
            explorer.model.config.audio_encoder.sampling_rate,
            format='wav'
        )
        buffer.seek(0)
        
        with open(filepath, 'wb') as f:
            f.write(buffer.getvalue())
            
        return jsonify({
            'audio_url': f'/static/interpolated.wav?t={np.random.randint(10000)}'
        })
        
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    explorer = LatentSpaceExplorer(n_points=16)
    app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False)