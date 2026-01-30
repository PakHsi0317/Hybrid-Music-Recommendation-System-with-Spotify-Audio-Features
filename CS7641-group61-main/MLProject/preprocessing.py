import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix, coo_matrix
import warnings
warnings.filterwarnings('ignore')

class MusicDatasetPreprocessor:
    """
    Preprocessing pipeline for merging Million Song Dataset, Spotify features,
    and This Is My Jam data for ML model training.
    """
    
    def __init__(self, play_count_threshold: int = 3, data_dir: str = "./data"):
        self.play_count_threshold = play_count_threshold
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.msd_triplets = None
        self.spotify_features = None
        self.thisismyjam_data = None
        self.acousticbrainz_mapping = None
        self.final_dataset = None
        self.final_dataset_before_negatives = None
        self.user_song_matrix = None
        self.mappings = None
        
    
    def load_msd_triplets(self, filepath: str = None):
        """
        Load Million Song Dataset taste profile triplets.
        Format: user_id, song_id, play_count
        """
        print("\n=== Loading MSD Triplets ===")
        
        if filepath is None:
            filepath = self.data_dir / "train_triplets.txt"
            
        if not Path(filepath).exists():
            print(f"MSD triplets file not found at {filepath}")
            return None
        
        self.msd_triplets = pd.read_csv(
            filepath,
            sep='\t',
            header=None,
            names=['user_id', 'song_id', 'play_count']
        )
        
        print(f"Loaded {len(self.msd_triplets):,} triplets")
        print(f"Unique users: {self.msd_triplets['user_id'].nunique():,}")
        print(f"Unique songs: {self.msd_triplets['song_id'].nunique():,}")
        
        self.msd_triplets['liked'] = (
            self.msd_triplets['play_count'] >= self.play_count_threshold
        ).astype(int)
        
        print(f"Liked songs (play_count >= {self.play_count_threshold}): "
              f"{self.msd_triplets['liked'].sum():,}")
        
        return self.msd_triplets
    
    def load_spotify_features(self, filepath: str = None):
        """
        Load Kaggle Spotify dataset with audio features.
        """
        print("\n=== Loading Spotify Features ===")
        
        if filepath is None:
            filepath = self.data_dir / "spotify_data.csv"
        
        if not Path(filepath).exists():
            print(f"Spotify features file not found at {filepath}")
            return None
        
        self.spotify_features = pd.read_csv(filepath)
        
        if 'id' in self.spotify_features.columns:
            self.spotify_features.rename(columns={'id': 'spotify_id'}, inplace=True)
        
        print(f"Loaded {len(self.spotify_features):,} Spotify tracks")
        print(f"Features: {list(self.spotify_features.columns)}")
        
        return self.spotify_features
    
    def load_thisismyjam_data(self, jams_path: str = None, likes_path: str = None):
        """
        Load This Is My Jam dataset from TSV files.
        """
        print("\n=== Loading This Is My Jam Data ===")
        
        if jams_path is None:
            jams_path = self.data_dir / "jams.tsv"
        
        if not Path(jams_path).exists():
            print(f"Jams file not found at {jams_path}")
            return None
        
        jams = pd.read_csv(
            jams_path, 
            sep='\t',
            quoting=3,  
            engine='python',  
            on_bad_lines='skip' 
        )
        
        jams.columns = jams.columns.str.strip()
        
        print(f"Loaded {len(jams):,} jams")
        print(f"Columns: {list(jams.columns)}")
        
    
        if likes_path is None:
            likes_path = self.data_dir / "likes.tsv"
        
        
        likes = pd.read_csv(likes_path, sep='\t', engine='python')
        likes.columns = likes.columns.str.strip()
        print(f"Loaded {len(likes):,} likes")
        print(f"Columns: {list(likes.columns)}")
        
        jams_with_likes = jams.merge(
            likes,
            on='jam_id',
            how='left',
            suffixes=('_creator', '_liker')
        )
        print(f"Total jam-user interactions: {len(jams_with_likes):,}")
        
        self.thisismyjam_data = jams_with_likes
        return self.thisismyjam_data
    
    def extract_jam_preferences(self):
        """
        Extract user preferences from This Is My Jam data.
        Returns: DataFrame with (user_id, spotify_id, liked=1)
        """
        print("\n=== Extracting Jam Preferences ===")
        
        if self.thisismyjam_data is None:
            return pd.DataFrame(columns=['user_id', 'spotify_id', 'liked'])
        
        preferences = []
        
        spotify_jams = self.thisismyjam_data[
            self.thisismyjam_data['spotify_uri'].notna()
        ].copy()
        
        # (format: spotify:track:ID)
        spotify_jams['spotify_id'] = spotify_jams['spotify_uri'].str.split(':').str[-1]
         
        creator_prefs = spotify_jams[["user_id_creator", 'spotify_id']].copy()
        creator_prefs.columns = ['user_id', 'spotify_id']
        creator_prefs = creator_prefs[creator_prefs['user_id'].notna()]
        creator_prefs['liked'] = 1
        creator_prefs['source'] = 'creator'
        preferences.append(creator_prefs)
        print(f"Jam creators: {len(creator_prefs):,} preferences")
        
        # users who liked jams also liked the song
        if 'user_id_liker' in spotify_jams.columns:
            liker_prefs = spotify_jams[
                spotify_jams['user_id_liker'].notna()
            ][['user_id_liker', 'spotify_id']].copy()
            liker_prefs.columns = ['user_id', 'spotify_id']
            liker_prefs['liked'] = 1
            liker_prefs['source'] = 'liker'
            preferences.append(liker_prefs)
            print(f"Jam likers: {len(liker_prefs):,} preferences")
        
        if not preferences:
            print("No valid preferences extracted")
            return pd.DataFrame(columns=['user_id', 'spotify_id', 'liked'])
        
        all_prefs = pd.concat(preferences, ignore_index=True)
        all_prefs = all_prefs[['user_id', 'spotify_id', 'liked']].drop_duplicates()
        
        print(f"Total unique user-song preferences: {len(all_prefs):,}")
        
        return all_prefs
    
    
    def merge_datasets(self):
        """
        Merge Jam datased with Spotify features
        """
        if self.thisismyjam_data is None or self.spotify_features is None:
            print("Required datasets not loaded")
            return None
        
        jam_prefs = self.extract_jam_preferences()
        
        if len(jam_prefs) == 0:
            print("No jam preferences extracted")
            return None
        
        print("\n=== Merging Jams with Spotify ===")
        
        merged = jam_prefs.merge(
            self.spotify_features,
            on='spotify_id',
            how='inner'
        )
        
        print(f"Entries merged with Spotify features: {len(merged):,}")
        match_rate = len(merged)/len(jam_prefs)*100

        self.final_dataset_before_negatives = merged
        output_path = self.data_dir / "processed_music_data_no_negatives.csv"
        self.save_processed_data(self.final_dataset_before_negatives, output_path=output_path)
        self.create_sparse_matrix(final_dataset=merged)

        merged = self.generate_negative_interactions(merged)
        
        
        self.final_dataset = merged

        print("\n=== Final dataset summary ===")
        print(f"Total records: {len(self.final_dataset):,}")
        print(f"Unique users: {self.final_dataset['user_id'].nunique():,}")
        print(f"Unique songs: {self.final_dataset['spotify_id'].nunique():,}")
        print(f"Liked interactions: {self.final_dataset['liked'].sum():,}")
        print(f"Not liked interactions: {(self.final_dataset['liked']==0).sum():,}")

        self.save_processed_data(self.final_dataset)
        
        return self.final_dataset
    
    def generate_negative_interactions(self, merged = None):
        """
        Generate negative interactions for supervised learning
        """
        print("\n=== Generating Negative Examples ===")
        
        unique_users = merged['user_id'].unique()
        all_songs = self.spotify_features['spotify_id'].values
        
        positive_pairs = set(zip(merged['user_id'], merged['spotify_id']))

        user_positive_counts = merged.groupby('user_id').size()
        
        negative_users = []
        negative_songs = []
        
        for user in unique_users:
            # Number of negative samples (capped at 10)
            n_positives = user_positive_counts.get(user, 1)
            n_samples = min(n_positives, len(all_songs) - n_positives, 10)
            
            if n_samples <= 0:
                continue
            
            sampled_songs = np.random.choice(all_songs, size=n_samples * 3, replace=False)
            
            valid_negatives = [song for song in sampled_songs 
                             if (user, song) not in positive_pairs][:n_samples]
            
            negative_users.extend([user] * len(valid_negatives))
            negative_songs.extend(valid_negatives)
        
        if negative_users:
            negatives = pd.DataFrame({
                'user_id': negative_users,
                'spotify_id': negative_songs,
                'liked': 0
            })
            
            negatives = negatives.merge(
                self.spotify_features,
                on='spotify_id',
                how='inner'
            )
            
            print(f"Generated {len(negatives):,} negative examples")
            
            merged = pd.concat([merged, negatives], ignore_index=True)

        return merged
    
    def save_processed_data(self, data, output_path: str = None):
        """Save the processed dataset."""
        if output_path is None:
            output_path = self.data_dir / "processed_music_data.csv"
        
        if data is not None:
            data.to_csv(output_path, index=False)
            print(f"\nSaved processed data to {output_path}")
        else:
            print("No processed data to save")
    
    def load_processed_data(self, input_path: str = None):
        """
        Load previously processed dataset from CSV file.
        """
        if input_path is None:
            input_path = self.data_dir / "processed_music_data.csv"
        
        if not Path(input_path).exists():
            print(f"Processed data file not found at {input_path}")
            return None
        
        print(f"\n=== Loading Processed Data ===")
        
        self.final_dataset = pd.read_csv(input_path)
        return self.final_dataset
    
    def create_sparse_matrix(self, final_dataset = None):
        """
        Create a sparse user-song matrix from the final dataset.
        """
        print("\n=== Creating Sparse User-Song Matrix ===")
        
        if final_dataset is None:
            final_dataset = self.final_dataset

        if final_dataset is None:
            print("No final dataset available. Run merge_datasets() first.")
            return None
    
        data = final_dataset[final_dataset['liked'].notna()].copy()
        
        unique_users = data['user_id'].unique()
        unique_songs = data['spotify_id'].unique()
        
        user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        song_to_idx = {song: idx for idx, song in enumerate(unique_songs)}
        
        idx_to_user = {idx: user for user, idx in user_to_idx.items()}
        idx_to_song = {idx: song for song, idx in song_to_idx.items()}
        
        print(f"Users: {len(unique_users):,}")
        print(f"Songs: {len(unique_songs):,}")
        print(f"Matrix shape: ({len(unique_users):,}, {len(unique_songs):,})")
        print(f"Sparsity: {(1 - len(data) / (len(unique_users) * len(unique_songs))) * 100:.4f}%")
        
        row_indices = data['user_id'].map(user_to_idx).values
        col_indices = data['spotify_id'].map(song_to_idx).values
        values = data['liked'].values
        

        sparse_matrix = coo_matrix(
            (values, (row_indices, col_indices)),
            shape=(len(unique_users), len(unique_songs))
        )
        
        sparse_matrix = sparse_matrix.tocsr()

        self.user_song_matrix = sparse_matrix
        self.mappings = user_to_idx, idx_to_user, song_to_idx, idx_to_song
        
        return sparse_matrix, user_to_idx, idx_to_user, song_to_idx, idx_to_song
       



