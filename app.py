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
from functools import lru_cache
import threading
from queue import Queue
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)

class LatentSpaceExplorer:
    def __init__(self, n_points=16, save_dir='static', cache_size=32):
        self.n_points = n_points
        self.save_dir = save_dir
        self.points_file = os.path.join(save_dir, 'audio_points.pkl')
        self.mapping_file = os.path.join(save_dir, 'latent_mapping.json')
        self.cache_size = cache_size
        
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Setup device and optimize for inference
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        logger.info("Loading model...")
        self.model = MusicgenForConditionalGeneration.from_pretrained(
            "facebook/musicgen-small",
            torch_dtype=torch.float16 if self.device.type != "cpu" else torch.float32
        ).to(self.device)
        self.model.eval()
        
        # Enable model optimizations
        if hasattr(self.model, 'half') and self.device.type != "cpu":
            self.model.half()  # Convert to FP16
        torch.backends.cudnn.benchmark = True
        
        self.processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        
        # Initialize caches
        self.latent_cache = {}
        self.audio_cache = {}
        self.interpolation_cache = {}
        
        # Setup background generation queue
        self.generation_queue = Queue()
        self.background_thread = threading.Thread(target=self._background_generation, daemon=True)
        self.background_thread.start()
        
        # Load or generate points
        self.audio_points = self.load_or_generate_points()
        
        # Create latent mapping
        self.latent_mapping = self.create_latent_mapping()
        
        # Pre-generate common interpolations
        self._pregenerate_interpolations()
        
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

    @torch.inference_mode()
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

    @torch.inference_mode()
    def _pregenerate_interpolations(self):
        """Pre-generate interpolations for common positions"""
        logger.info("Pre-generating interpolations...")
        grid_points = 4  # 4x4 grid
        for x in np.linspace(0, 1, grid_points):
            for y in np.linspace(0, 1, grid_points):
                self.generation_queue.put((x, y))

    def _background_generation(self):
        """Background thread for generating audio"""
        while True:
            try:
                x, y = self.generation_queue.get()
                key = f"{x:.2f}_{y:.2f}"
                if key not in self.interpolation_cache:
                    audio = self._generate_at_position(x, y)
                    self.interpolation_cache[key] = audio
                    
                    # Limit cache size
                    if len(self.interpolation_cache) > self.cache_size:
                        self.interpolation_cache.pop(next(iter(self.interpolation_cache)))
            except Exception as e:
                logger.error(f"Background generation error: {e}")
            finally:
                self.generation_queue.task_done()

    @torch.inference_mode()
    def _generate_at_position(self, x, y):
        """Optimized generation at position"""
        positions = self.latent_mapping['umap']
        distances = {
            point_id: np.sqrt((x - pos['x'])**2 + (y - pos['y'])**2)
            for point_id, pos in positions.items()
        }
        nearest_points = sorted(distances.items(), key=lambda x: x[1])[:4]
        
        # Compute weights once
        total_weight = sum(1/d for _, d in nearest_points)
        weights = [1/d/total_weight for _, d in nearest_points]
        
        # Batch process the latents
        latents = [self.latent_cache[point_id].to(self.device) for point_id, _ in nearest_points]
        
        # Efficient interpolation
        result_scores = latents[0]
        accumulated_weight = weights[0]
        
        for i in range(1, len(nearest_points)):
            if accumulated_weight >= 0.99:  # Optimization: early exit
                break
            
            relative_weight = weights[i] / (accumulated_weight + weights[i])
            result_scores = torch.lerp(result_scores, latents[i], relative_weight)
            accumulated_weight += weights[i]
        
        # Optimize generation
        inputs = self.processor(
            text=["interpolated sound"],
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        audio = self.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            guidance_scale=3.0
        )[0]
        
        return audio.cpu()

    @lru_cache(maxsize=32)
    def get_cached_position(self, x, y):
        """Get cached interpolation result"""
        key = f"{x:.2f}_{y:.2f}"
        return self.interpolation_cache.get(key)

    def interpolate_position(self, x, y, method='umap'):
        """Interpolate with caching and background generation"""
        # Check cache first
        cached_audio = self.get_cached_position(x, y)
        if cached_audio is not None:
            return cached_audio
            
        # Generate in background for future requests
        self.generation_queue.put((x, y))
        
        # Find nearest cached point
        nearest_cache = min(
            self.interpolation_cache.items(),
            key=lambda item: np.sqrt(
                (float(item[0].split('_')[0]) - x)**2 + 
                (float(item[0].split('_')[1]) - y)**2
            ),
            default=(None, None)
        )
        
        if nearest_cache[1] is not None:
            return nearest_cache[1]
            
        # If no cache available, generate directly
        return self._generate_at_position(x, y)

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
        
        filepath = os.path.join(explorer.save_dir, 'interpolated.wav')
        buffer = io.BytesIO()
        
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