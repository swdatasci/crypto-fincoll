import sys
sys.path.insert(0, '/home/rford/caelum/caelum-supersystem/finvec')
from fincoll.inference.prediction_engine import PredictionEngine

print('Loading model...')
engine = PredictionEngine(
    checkpoint_path='models/finvec_continuous_10epoch.pt',
    device='cpu'
)
print(f'Model loaded! use_continuous_horizons: {engine.use_continuous_horizons}')
