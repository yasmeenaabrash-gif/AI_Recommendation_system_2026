import pandas as pd
import numpy as np

# --- Phase 1: Data Initialization ---
def load_and_merge_data():
    try:
        # Loading datasets from excel files
        users = pd.read_excel('users.xlsx')
        products = pd.read_excel('products.xlsx')
        ratings = pd.read_excel('ratings.xlsx')
        behavior = pd.read_excel('behavior.xlsx')

        # Merging datasets to create a unified data structure
        merged_step1 = pd.merge(behavior, users, on='user_id')
        final_dataset = pd.merge(merged_step1, products, on='product_id')
        
        return final_dataset
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# --- Phase 2: Fitness Function Definition ---
def calculate_fitness(df):
    """
    Calculates the fitness score for each user-product interaction.
    Weights: Purchased = 5, Clicked = 3, Viewed = 1
    """
    df['fitness_score'] = (
        (df['purchased'] * 5) + 
        (df['clicked'] * 3) + 
        (df['viewed'] * 1)
    )
    return df

# --- Phase 3: Genetic Algorithm Core Logic ---
def run_genetic_algorithm(data):
    # 1. Selection Phase: Identifying the 'Elite' recommendations
    # Selecting interactions with the highest fitness scores (top 10%)
    threshold = data['fitness_score'].quantile(0.9)
    elite_population = data[data['fitness_score'] >= threshold]

    # 2. Crossover Phase (Optimization):
    # Extracting the most successful product categories to form the next generation
    optimized_categories = elite_population['category'].value_counts().head(5).index.tolist()
    
    # 3. Mutation Phase (Diversity):
    # Introducing a random category to ensure the system explores new interests
    all_categories = data['category'].dropna().unique()
    mutation_gene = np.random.choice(all_categories)
    
    return optimized_categories, mutation_gene

# --- Main Execution ---
if __name__ == "__main__":
    # Execute loading and merging
    dataset = load_and_merge_data()
    
    if dataset is not None:
        # Step 1: Calculate fitness scores
        dataset = calculate_fitness(dataset)
        print("✅ Data Pipeline: Success. Fitness scores calculated.")

        # Step 2: Apply Genetic Algorithm
        best_categories, mutation = run_genetic_algorithm(dataset)
        
        print("\n" + "="*30)
        print("GENETIC ALGORITHM RESULTS")
        print("="*30)
        print(f"Optimized Categories (Crossover): {best_categories}")
        print(f"Random Exploration (Mutation): {mutation}")
        print("="*30)