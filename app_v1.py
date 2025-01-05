import torch
import torchaudio
import numpy as np
from flask import Flask, render_template, request, jsonify
import io
import os
import logging
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import pickle
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)

class LatentSpaceExplorer:
    def __init__(self, n_points=16, save_dir='static'):
        self.n_points = n_points
        self.save_dir = save_dir
        self.points_file = os.path.join(save_dir, 'audio_points.pkl')
        self.backup_file = os.path.join(save_dir, 'audio_points.pkl.backup')
        
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        self.device = torch.device("cpu")
        self.mps_device = torch.device("mps")
        
        logger.info("Loading model...")
        self.model = MusicgenForConditionalGeneration.from_pretrained(
            "facebook/musicgen-small",
            torch_dtype=torch.float32
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        
        # Try to load existing points with fallback
        self.audio_points = self.load_or_generate_points()
        logger.info(f"Initialized with {len(self.audio_points)} points")

    def load_or_generate_points(self):
        """Try to load points with fallback to generation"""
        try:
            if os.path.exists(self.points_file):
                try:
                    points = self.load_points(self.points_file)
                    if self.verify_points(points):
                        logger.info("Successfully loaded points from main file")
                        return points
                except Exception as e:
                    logger.warning(f"Error loading main points file: {e}")
                    
            if os.path.exists(self.backup_file):
                try:
                    points = self.load_points(self.backup_file)
                    if self.verify_points(points):
                        logger.info("Successfully loaded points from backup file")
                        return points
                except Exception as e:
                    logger.warning(f"Error loading backup points file: {e}")
            
            # If we get here, generate new points
            logger.info("Generating new points...")
            points = self.generate_initial_points()
            self.save_points(points)
            return points
            
        except Exception as e:
            logger.error(f"Error in load_or_generate_points: {e}")
            return self.generate_initial_points()

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

    def save_points(self, points):
        """Save points with backup"""
        try:
            # Convert tensors to CPU before saving
            points_cpu = {k: v.cpu() for k, v in points.items()}
            
            # If main file exists, make it the backup
            if os.path.exists(self.points_file):
                shutil.copy2(self.points_file, self.backup_file)
            
            # Save new points
            with open(self.points_file, 'wb') as f:
                pickle.dump(points_cpu, f)
            
            logger.info("Successfully saved points")
            return True
        except Exception as e:
            logger.error(f"Error saving points: {e}")
            return False

    def load_points(self, filepath):
        """Load points from file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def generate_initial_points(self):
        """Generate initial audio points"""
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
                
                # Save audio file
                filepath = os.path.join(self.save_dir, f"point_{i}.wav")
                self.save_audio(audio[0], filepath)
                points[f"point_{i}"] = audio[0]
        
        return points

    def save_audio(self, tensor, filepath):
        """Save audio tensor to file"""
        tensor = tensor / torch.max(torch.abs(tensor))
        if len(tensor.shape) == 1:
            tensor = tensor.unsqueeze(0)
        torchaudio.save(filepath, tensor, self.model.config.audio_encoder.sampling_rate)

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

if __name__ == '__main__':
    explorer = LatentSpaceExplorer(n_points=16)
    app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False)