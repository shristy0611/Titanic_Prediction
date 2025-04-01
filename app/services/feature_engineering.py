import pandas as pd
import numpy as np

class FeatureEngineer:
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"Input DataFrame columns: {df.columns.tolist()}")  # Debug print
        
        df = df.copy()
        
        # Normalize column names to handle case sensitivity
        df.columns = [col.lower() for col in df.columns]
        print(f"After lowercase: {df.columns.tolist()}")  # Debug print
        
        # Ensure correct column names regardless of input casing
        df = df.rename(columns={'sibsp': 'sibsp', 'parch': 'parch'})
        
        # Family size
        df['familysize'] = df['sibsp'] + df['parch'] + 1
        df['isalone'] = (df['familysize'] == 1).astype(int)
        
        # Age imputation
        if 'age' not in df.columns:
            print(f"ERROR: 'age' column not found. Available columns: {df.columns.tolist()}")
            df['age'] = 30.0  # Default value if missing
        
        df['age'] = df['age'].fillna(df['age'].median())
        
        # Fare normalization
        if 'fare' not in df.columns:
            print(f"ERROR: 'fare' column not found. Available columns: {df.columns.tolist()}")
            df['fare'] = 30.0  # Default value if missing
            
        df['fare'] = df['fare'].fillna(df['fare'].median())
        df['fare'] = np.log1p(df['fare'])

        # Fill missing Embarked with the mode
        if 'embarked' not in df.columns:
            print(f"ERROR: 'embarked' column not found. Available columns: {df.columns.tolist()}")
            df['embarked'] = 'S'  # Default value if missing
            
        mode_embarked = df['embarked'].mode()[0]
        df['embarked'] = df['embarked'].fillna(mode_embarked)

        # One-hot encode categorical features
        categorical_cols = ['sex', 'embarked']
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dummy_na=False) # drop_first to avoid multicollinearity

        # Clean up column names
        df.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in df.columns]
        print(f"After cleanup: {df.columns.tolist()}")  # Debug print
        
        # Rename columns to match expected model feature names
        # We use a capitalized version to match MODEL_FEATURES in config.py
        column_mapping = {
            'age': 'Age', 
            'fare': 'Fare',
            'familysize': 'FamilySize',
            'pclass': 'Pclass',
            'isalone': 'IsAlone',
            'sex_male': 'Sex_male',
            'embarked_q': 'Embarked_Q',
            'embarked_s': 'Embarked_S'
        }
        df = df.rename(columns=column_mapping)
        print(f"Final columns: {df.columns.tolist()}")  # Debug print
            
        return df
