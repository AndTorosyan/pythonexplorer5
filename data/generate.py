"""
Dataset Generation
==================
Այս սկրիպտը ստեղծում է 2D "moons" տվյալների բազա binary classification-ի համար
և պահպանում այն որպես CSV ֆայլ:
"""
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

def generate_dataset(n_samples=1000, noise=0.2, random_state=42):
    """
    Գեներացնել 2D moons տվյալների բազա:
    """
    # Քայլ 1: Գեներացնել լուսինները
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

    # Քայլ 2: Ստանդարտացնել հատկանիշները (Standardize features)
    # StandardScaler-ը տվյալները կենտրոնացնում է 0-ի շուրջ՝ 1 ստանդարտ շեղմամբ
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Քայլ 3: Ստեղծել DataFrame
    # TODO: Ստեղծել pandas DataFrame 'x1', 'x2', 'label' սյուներով
    # Հուշում: X-ն ունի (1000, 2) ձև — սյուն 0-ն x1-ն է, սյուն 1-ը՝ x2-ը
    df = pd.DataFrame(X, columns=['x1', 'x2', 'label'])

    return df

if __name__ == "__main__":
    df = generate_dataset()
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['label'].value_counts()}")

    
    # Հուշում: df.to_csv("path/to/file.csv", index=False)
    df.to_csv("data/dataset.csv", index=False)
    print("Dataset saved to data/dataset.csv")