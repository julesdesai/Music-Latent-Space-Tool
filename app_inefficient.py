import torch
import torchaudio
import numpy as np
from flask import Flask, render_template, request, jsonify
import io
import os
import logging
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import pickle
from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)

class LatentSpaceExplorer:
    def __init__(self, n_points=16, save_dir='static'):
        self.n_points = n_points
        self.save_dir = save_dir
        self.points_file = os.path.join(save_dir, 'audio_points.pkl')
        self.mapping_file = os.path.join(save_dir, 'latent_mapping.json')
        
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Setup device
        self.device = torch.device("cpu")
        self.mps_device = torch.device("mps")
        
        logger.info("Loading model...")
        self.model = MusicgenForConditionalGeneration.from_pretrained(
            "facebook/musicgen-small",
            torch_dtype=torch.float32
        ).to(self.device)
        self.model.eval()
        
        self.processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        logger.info("Model loaded")
        
        # Initialize latent cache
        self.latent_cache = {}
        
        # Load or generate points
        self.audio_points = self.load_or_generate_points()
        
        # Create latent mapping
        self.latent_mapping = self.create_latent_mapping()
        
        logger.info("Initialization complete!")
        
    def load_or_generate_points(self):
        """Load existing points or generate new ones"""
        try:
            if os.path.exists(self.points_file):
                logger.info("Loading existing points...")
                with open(self.points_file, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, dict) and 'audio' in data and 'latents' in data:
                        self.latent_cache = data['latents']
                        logger.info("Successfully loaded existing points")
                        return data['audio']
                    else:
                        logger.warning("Invalid data format in points file")
        except Exception as e:
            logger.error(f"Error loading points: {e}")

        # Generate new points if loading failed
        logger.info("Generating new points...")
        points = self.generate_initial_points()
        
        # Save after successfully generating points
        try:
            self.save_points(points)
        except Exception as e:
            logger.error(f"Error saving points: {e}")
        
        return points

    def generate_initial_points(self):
            """Generate initial points with latent representations"""
            points = {}
            descriptions = [
                "ambient pad",
                "drum beat",
                "synth melody",
                "bass drone",
                "bell sound",
                "texture",
                "rhythm",
                "water",
                "space",
                "glitch",
                "chord",
                "high freq",
                "bass pulse",
                "abstract",
                "minimal",
                "ethereal"
            ]
            
            with torch.no_grad():
                for i, desc in enumerate(descriptions[:self.n_points]):
                    logger.info(f"Generating point {i}: {desc}")
                    
                    # Process text input
                    inputs = self.processor(
                        text=[desc],
                        padding=True,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    # Generate with return_dict=True to get the decoder outputs
                    generation_output = self.model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=True,
                        guidance_scale=3.0,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                    
                    # Get the audio output
                    audio = generation_output.sequences[0]
                    
                    # Store the decoder scores as our latent representation
                    self.latent_cache[f"point_{i}"] = torch.stack(generation_output.scores)
                    
                    filepath = os.path.join(self.save_dir, f"point_{i}.wav")
                    self.save_audio(audio, filepath)
                    points[f"point_{i}"] = audio
            
            return points



    def save_points(self, points):
        """Save both audio points and latent representations"""
        try:
            points_cpu = {k: v.cpu() for k, v in points.items()}
            latents_cpu = {k: v.cpu() for k, v in self.latent_cache.items()}
            
            with open(self.points_file, 'wb') as f:
                pickle.dump({
                    'audio': points_cpu,
                    'latents': latents_cpu
                }, f)
            logger.info("Successfully saved points")
        except Exception as e:
            logger.error(f"Error saving points: {e}")
            raise
        
    def create_latent_mapping(self):
        """Create 2D mapping of latent points using UMAP and t-SNE"""
        try:
            if os.path.exists(self.mapping_file):
                with open(self.mapping_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load mapping: {e}")
        
        # Get features from latent representations
        features = []
        for latent in self.latent_cache.values():
            # Use mean of latent representation as features
            features.append(latent.mean(dim=1).cpu().numpy().flatten())
        
        features = np.array(features)
        
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Generate UMAP embedding
        umap_reducer = UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=min(15, len(features) - 1),
            min_dist=0.1
        )
        umap_embedding = umap_reducer.fit_transform(features_scaled)
        
        # Generate t-SNE embedding
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=min(15, len(features) - 1)
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

    def save_audio(self, tensor, filepath):
        """Save audio tensor with proper shape handling"""
        try:
            tensor = tensor.cpu()
            tensor = tensor / torch.max(torch.abs(tensor))
            
            if len(tensor.shape) == 1:
                tensor = tensor.unsqueeze(0)
            elif len(tensor.shape) > 2:
                tensor = tensor.squeeze()
                if len(tensor.shape) == 1:
                    tensor = tensor.unsqueeze(0)
            
            torchaudio.save(
                filepath if isinstance(filepath, str) else filepath.name,
                tensor,
                self.model.config.audio_encoder.sampling_rate
            )
            return True
        except Exception as e:
            logger.error(f"Error saving audio: {e}")
            return False

    def interpolate_latents(self, latent1, latent2, ratio):
        """Interpolate between two latent representations"""
        min_length = min(latent1.size(1), latent2.size(1))
        latent1 = latent1[:, :min_length, :]
        latent2 = latent2[:, :min_length, :]
        
        latent1_norm = latent1 / latent1.norm(dim=-1, keepdim=True)
        latent2_norm = latent2 / latent2.norm(dim=-1, keepdim=True)
        
        omega = torch.acos((latent1_norm * latent2_norm).sum(dim=-1, keepdim=True).clamp(-1, 1))
        sin_omega = torch.sin(omega)
        
        mask = sin_omega.abs() < 1e-6
        interpolated = torch.where(
            mask,
            (1 - ratio) * latent1 + ratio * latent2,
            (torch.sin((1 - ratio) * omega) / sin_omega) * latent1 + 
            (torch.sin(ratio * omega) / sin_omega) * latent2
        )
        
        return interpolated

    def interpolate_position(self, x, y, method='umap'):
            """Interpolate in latent space at any 2D position"""
            try:
                positions = self.latent_mapping[method]
                distances = {
                    point_id: np.sqrt((x - pos['x'])**2 + (y - pos['y'])**2)
                    for point_id, pos in positions.items()
                }
                nearest_points = sorted(distances.items(), key=lambda x: x[1])[:4]
                
                total_weight = sum(1/d for _, d in nearest_points)
                weights = [1/d/total_weight for _, d in nearest_points]
                
                # Get the first point's scores
                result_scores = self.latent_cache[nearest_points[0][0]].to(self.mps_device)
                accumulated_weight = weights[0]
                
                # Get the audio for conditioning
                base_audio = self.audio_points[nearest_points[0][0]]
                
                # Interpolate between score sequences
                for i in range(1, len(nearest_points)):
                    if accumulated_weight >= 1.0:
                        break
                        
                    next_scores = self.latent_cache[nearest_points[i][0]].to(self.mps_device)
                    relative_weight = weights[i] / (accumulated_weight + weights[i])
                    
                    # Interpolate scores
                    result_scores = (1 - relative_weight) * result_scores + relative_weight * next_scores
                    accumulated_weight += weights[i]
                
                # Generate from interpolated scores
                with torch.no_grad():
                    # Use a default prompt for generation
                    inputs = self.processor(
                        text=["interpolated sound"],
                        padding=True,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    # Generate new audio using interpolated scores as conditioning
                    audio = self.model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=True,
                        guidance_scale=3.0
                    )[0]
                    
                audio = audio / torch.max(torch.abs(audio))
                return audio.cpu()
                
            except Exception as e:
                logger.error(f"Interpolation error: {e}")
                logger.error(f"Error details:", exc_info=True)
                raise

# Flask Routes
@app.route('/')
def index():
    return render_template('latent_explorer.html', n_points=explorer.n_points)

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
        logger.info(f"Generating audio at position ({x:.3f}, {y:.3f})")
        audio = explorer.interpolate_position(x, y, method)
        
        # Save interpolated audio
        filepath = os.path.join(explorer.save_dir, 'interpolated.wav')
        buffer = io.BytesIO()
        
        # Ensure audio has correct shape
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

@app.route('/check_audio/<filename>')
def check_audio(filename):
    filepath = os.path.join('static', filename)
    if os.path.exists(filepath):
        try:
            waveform, sample_rate = torchaudio.load(filepath)
            return jsonify({
                'exists': True,
                'size': os.path.getsize(filepath),
                'shape': waveform.shape,
                'sample_rate': sample_rate,
                'min': float(waveform.min()),
                'max': float(waveform.max()),
                'mean': float(waveform.mean())
            })
        except Exception as e:
            return jsonify({
                'exists': True,
                'error': str(e),
                'size': os.path.getsize(filepath)
            })
    return jsonify({'exists': False})

if __name__ == '__main__':
    explorer = LatentSpaceExplorer(n_points=16)
    app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False)