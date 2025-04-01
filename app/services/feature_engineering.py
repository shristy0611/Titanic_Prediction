import pandas as pd
import numpy as np

class FeatureEngineer:
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Extract title from name
        df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        
        # Family size
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        # Age imputation
        df['Age'] = df['Age'].fillna(df['Age'].median())
        
        # Fare normalization
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        df['Fare'] = np.log1p(df['Fare'])
        
        return df
