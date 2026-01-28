import pandas as pd
import numpy as np
import joblib
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset_to_train.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

class WarSimulator:
    def __init__(self, data_path=None):
        print("--- Initializing War Room ---")
        # 1. Load the latest "snapshot" of the war
        if data_path is None:
            data_path = DATA_PATH
        self.df = pd.read_csv(data_path, parse_dates=['date'])
        self.last_date = self.df['date'].max()
        print(f"Data updated until: {self.last_date}")

        # 2. Load the 3 Brains (Models)
        print("Loading Tactical AI...")
        self.model_A = joblib.load(os.path.join(MODELS_DIR, 'model_A_front_dynamics.pkl'))  # Movement
        self.model_B = joblib.load(os.path.join(MODELS_DIR, 'model_B_encirclement.pkl'))    # Encirclement
        self.model_C = joblib.load(os.path.join(MODELS_DIR, 'model_C_capture.pkl'))         # Capture
        
        # Get expected feature names from each model to ensure order
        self.feats_A = self.model_A.get_booster().feature_names
        self.feats_B = self.model_B.get_booster().feature_names
        self.feats_C = self.model_C.get_booster().feature_names

    def get_city_state(self, city_name):
        """Finds the most recent state of a city."""
        city_data = self.df[self.df['name'] == city_name]
        if city_data.empty:
            raise ValueError(f"City '{city_name}' not found.")
        
        # Return the last known row (the 'today')
        return city_data.sort_values('date').iloc[-1].copy()

    def run_simulation(self, city_name, days=60, scenario='inertial'):
        """
        Executes the recursive projection.
        """
        # Configure scenario multiplier
        modifiers = {'conservative': 0.5, 'inertial': 1.0, 'aggressive': 1.5}
        modifier = modifiers.get(scenario, 1.0)
        
        print(f"\n>>> SIMULATING: {city_name} | Scenario: {scenario.upper()} (x{modifier}) | {days} Days")
        
        current_state = self.get_city_state(city_name)
        history = []
        
        # If the city is already occupied, no simulation needed
        if current_state['is_captured'] == 1:
            print("WARNING: This city is already occupied. Simulation aborted.")
            return None

        # --- SIMULATION LOOP (Day by Day) ---
        for day in range(1, days + 1):
            
            # 1. PREDICTION A: How much does the front move?
            # FIX: .astype(float) to avoid 'object' error
            input_A = current_state[self.feats_A].to_frame().T.astype(float)
            pred_delta = self.model_A.predict(input_A)[0]
            
            # APPLY THE SCENARIO
            adjusted_delta = pred_delta * modifier
            
            # Update the physical reality
            new_dist = max(0, current_state['dist_to_front_m'] - adjusted_delta)
            
            # Update Momentum
            current_state['delta_dist_daily'] = adjusted_delta
            current_state['momentum_7d'] = (current_state['momentum_7d'] * 0.9) + (adjusted_delta * 0.1)
            current_state['momentum_30d'] = (current_state['momentum_30d'] * 0.95) + (adjusted_delta * 0.05)
            
            # 2. PREDICTION B: How does this affect encirclement?
            current_state['dist_to_front_m'] = new_dist
            
            # FIX: .astype(float)
            input_B = current_state[self.feats_B].to_frame().T.astype(float)
            pred_encirclement = self.model_B.predict(input_B)[0]
            
            # Clip to maintain realism (0-1)
            pred_encirclement = np.clip(pred_encirclement, 0, 1)
            current_state['encirclement_score'] = pred_encirclement
            
            # 3. PREDICTION C: Does the city fall?
            # FIX: .astype(float)
            input_C = current_state[self.feats_C].to_frame().T.astype(float)
            
            # Use predict_proba
            prob_capture = self.model_C.predict_proba(input_C)[0][1]
            
            # Binary decision (50% threshold)
            is_captured = 1 if prob_capture > 0.5 else 0
            
            # Save to history
            history.append({
                'day': day,
                'dist_to_front': round(new_dist, 0),
                'delta_real': round(adjusted_delta, 1),
                'encirclement': round(pred_encirclement, 3),
                'prob_capture': round(prob_capture, 3),
                'status': 'OCCUPIED' if is_captured else 'FREE'
            })
            
            # If it falls, stop the simulation
            if is_captured:
                print(f"!!! CITY FALLS ON DAY {day} !!!")
                break
        
        return pd.DataFrame(history)

# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    sim = WarSimulator()
    
    # CHANGE THIS TO A CITY YOU'RE INTERESTED IN
    TARGET_CITY = "Краматорськ"  # Pokrovsk
    
    # Run the 3 scenarios
    results_conservative = sim.run_simulation(TARGET_CITY, days=60, scenario='conservative')
    results_inertial = sim.run_simulation(TARGET_CITY, days=60, scenario='inertial')
    results_aggressive = sim.run_simulation(TARGET_CITY, days=60, scenario='aggressive')
    
    print("\n--- RESULTS: Inertial Scenario (First 10 days) ---")
    if results_inertial is not None:
        print(results_inertial.head(10))
        print("\nFinal State (Day 60):")
        print(results_inertial.tail(1))