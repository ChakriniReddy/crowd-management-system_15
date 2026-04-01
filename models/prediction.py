import numpy as np
from sklearn.linear_model import LinearRegression
from collections import deque

class CrowdPredictor:
    def __init__(self, history_len=30, predict_steps=30, grid_threshold=3, total_threshold=10):
        self.history_len = history_len
        self.predict_steps = predict_steps
        self.grid_threshold = grid_threshold
        self.total_threshold = total_threshold
        
        self.history = {'total': deque(maxlen=history_len), 'grids': {}}
        
    def update_and_predict(self, current_counts):
        predictions = {'total': 0, 'grids': {}, 'overflows': []}
        growth_rates = {'total': 0, 'grids': {}}
        
        # Update Total
        self.history['total'].append(current_counts['total'])
        pred, rate = self._forecast(self.history['total'])
        predictions['total'] = pred
        growth_rates['total'] = rate
        
        if pred >= self.total_threshold:
            predictions['overflows'].append("Global")

        # Update Grids
        for grid_id, count in current_counts['grids'].items():
            if grid_id not in self.history['grids']:
                self.history['grids'][grid_id] = deque(maxlen=self.history_len)
            
            self.history['grids'][grid_id].append(count)
            pred, rate = self._forecast(self.history['grids'][grid_id])
            predictions['grids'][grid_id] = pred
            growth_rates['grids'][grid_id] = rate
            
            if pred >= self.grid_threshold:
                predictions['overflows'].append(f"Grid {grid_id}")
                
        return predictions, growth_rates

    def _forecast(self, data_queue):
        if len(data_queue) < 5:
            # Need some data points to fit a line
            return data_queue[-1], 0
        
        X = np.arange(len(data_queue)).reshape(-1, 1)
        y = np.array(data_queue)
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict future point
        future_X = np.array([[len(data_queue) + self.predict_steps]])
        pred_y = max(0, int(model.predict(future_X)[0]))
        
        # Rate of growth per step (slope)
        growth_rate = model.coef_[0]
        
        return pred_y, growth_rate
