import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import json
import os
import numpy as np
import pickle
from tqdm import tqdm
import gc
import warnings
from PIL import Image
import open_clip
import pandas as pd
import csv
import re
import time
from typing import List, Dict, Tuple
import ijson
from collections import defaultdict
import string

# BM25 implementation
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    print("Warning: rank_bm25 is not installed. Installing it for BM25 functionality...")
    try:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rank-bm25"])
        from rank_bm25 import BM25Okapi
        BM25_AVAILABLE = True
        print("rank_bm25 installed successfully.")
    except:
        print("Failed to install rank_bm25. Using manual BM25 implementation.")
        BM25_AVAILABLE = False

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Langchain for semantic chunking
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("Warning: Langchain is not installed. Falling back to a custom text splitter.")
    print("For better semantic chunking, please install it with: pip install langchain")
    LANGCHAIN_AVAILABLE = False

# NLTK for sentence tokenization
try:
    import nltk
    nltk.data.find('tokenizers/punkt')
except (ImportError, nltk.downloader.DownloadError):
    print("NLTK 'punkt' not found. Downloading...")
    nltk.download('punkt')
    print("'punkt' downloaded.")

try:
    import ijson
    IJSON_AVAILABLE = True
except ImportError:
    print("Warning: ijson is not installed. This is required for processing large JSON files.")
    print("Please install it with: pip install ijson")
    IJSON_AVAILABLE = False

# --- 1. CONFIGURATION ---

# Paths
DATA_JSON_PATH = '/root/EVENTA/data/database/database.json'
IMAGE_FOLDER_PATH = '/root/EVENTA/data/database_original_image/database_img'
QUERY_FILE_PATH = '/root/23tnt/Track2/Minh/query.csv'

# Output / Cache Paths
IMAGE_EMBEDDINGS_PATH = '/root/23tnt/Track2/image_embeddings/CLIPembeddings_bigG14.pkl'
CHUNK_EMBEDDINGS_PATH = 'nomic_chunk_embeddings.pkl'
BM25_INDEX_PATH = 'bm25_article_index.pkl'
SUBMISSION_CSV_PATH = 'submission_bm25.csv'

# Models
IMAGE_ENCODER_MODEL = 'ViT-bigG-14'
IMAGE_ENCODER_PRETRAINED = 'laion2b_s39b_b160k'
TEXT_EMBEDDER_MODEL = 'nomic-ai/nomic-embed-text-v1.5'

# Text Chunking Parameters
CHUNK_SIZE = 384  # Optimal for Nomic, focused embeddings
CHUNK_OVERLAP = 64 # In tokens

# Retrieval & Re-ranking Parameters
TOP_M_CHUNKS = 100 # Best 100
TOP_N_ARTICLES = 20 # Best 20
TOP_K_IMAGES = 10
TOP_K_CHUNKS_PER_ARTICLE = 5  # Number of best supporting chunks per article # Best 3

# Scoring weights for final image ranking
W1_HOLISTIC_VISUAL = 0.4   # Query-to-image visual similarity
W3_ARTICLE_SCORE = 0.6     # Article relevance score

# Supporting chunk modulator parameters
LAMBDA_CHUNK_BOOST = 0.2   # Multiplier strength for best chunk similarity boost
USE_CHUNK_MULTIPLIER = True  # Whether to use chunk similarity as multiplier

# Hybrid retrieval parameters
K_RRF = 60  # Constant for Reciprocal Rank Fusion
BM25_K1 = 1.2  # BM25 k1 parameter
BM25_B = 0.75  # BM25 b parameter

# --- 2. HELPER FUNCTIONS ---

def mean_pooling(model_output, attention_mask):
    """Mean pooling for Nomic embeddings."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def extract_image_id_from_path(image_path: str) -> str:
    """Extracts image ID (filename without extension) from a path."""
    try:
        return os.path.splitext(os.path.basename(image_path))[0]
    except Exception as e:
        print(f"Warning: Could not extract image ID from {image_path}: {e}")
        return ""

# --- 3. OFFLINE PREPROCESSING ---

class ImageEmbedder:
    """Handles embedding of all images using an OpenCLIP model."""
    def __init__(self, model_name, pretrained, device="cuda"):
        self.device = device
        print(f"Initializing ImageEmbedder on device: {self.device}")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        print("Image encoder (CLIP) loaded.")

    def create_and_save_embeddings(self, image_folder: str, output_path: str, batch_size: int = 64):
        """Finds all images, creates embeddings, and saves them to a file."""
        image_paths = [os.path.join(root, file) for root, _, files in os.walk(image_folder)
                       for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_paths:
            raise ValueError(f"No images found in {image_folder}")

        print(f"Found {len(image_paths)} images. Creating embeddings...")
        all_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(image_paths), batch_size), desc="Embedding Images"):
                batch_paths = image_paths[i:i+batch_size]
                images = [self.preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
                image_tensor = torch.stack(images).to(self.device)
                
                batch_embeddings = self.model.encode_image(image_tensor)
                batch_embeddings /= batch_embeddings.norm(dim=-1, keepdim=True)
                all_embeddings.append(batch_embeddings.cpu())

        embeddings_tensor = torch.cat(all_embeddings)
        
        # Save both embeddings and corresponding IDs
        image_ids = [extract_image_id_from_path(p) for p in image_paths]
        
        print(f"Saving {len(image_ids)} image embeddings to {output_path}...")
        with open(output_path, 'wb') as f:
            pickle.dump({'ids': image_ids, 'embeddings': embeddings_tensor.numpy()}, f)
        print("Image embeddings saved.")

class ArticleEmbedder:
    """Handles chunking and embedding of article text using Nomic."""
    def __init__(self, model_name, device="cuda"):
        self.device = device
        print(f"Initializing ArticleEmbedder on device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).eval().to(self.device)
        print("Text embedder (Nomic) loaded.")

        if LANGCHAIN_AVAILABLE:
            print("Using LangChain's RecursiveCharacterTextSplitter for semantic chunking.")
            # We define a custom length function using the Nomic tokenizer
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=lambda text: len(self.tokenizer.encode(text, add_special_tokens=False)),
                separators=["\n\n", "\n", ". ", " ", ""], # Prioritize splitting by paragraphs and sentences
            )
        else:
            print("Using custom fallback text splitter.")
            self.text_splitter = None # Will signal to use the fallback method

    def _custom_text_splitter(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Splits text into chunks of specified token size with overlap. (Fallback method)"""
        if not text: return []
        
        sentences = nltk.sent_tokenize(text)
        tokenized_sentences = [self.tokenizer.encode(s, add_special_tokens=False) for s in sentences]

        chunks = []
        current_chunk_tokens = []
        for i, sentence_tokens in enumerate(tokenized_sentences):
            # If adding the next sentence exceeds chunk size, finalize the current chunk
            if len(current_chunk_tokens) + len(sentence_tokens) > chunk_size and current_chunk_tokens:
                chunks.append(self.tokenizer.decode(current_chunk_tokens))
                
                # Start new chunk with overlap
                overlap_start_index = max(0, len(current_chunk_tokens) - chunk_overlap)
                current_chunk_tokens = current_chunk_tokens[overlap_start_index:]

            # If a single sentence is too big, split it aggressively
            if len(sentence_tokens) > chunk_size:
                if current_chunk_tokens: # Finalize pending chunk first
                    chunks.append(self.tokenizer.decode(current_chunk_tokens))
                    current_chunk_tokens = []
                
                for j in range(0, len(sentence_tokens), chunk_size - chunk_overlap):
                    sub_chunk = sentence_tokens[j:j + chunk_size]
                    chunks.append(self.tokenizer.decode(sub_chunk))
            else:
                current_chunk_tokens.extend(sentence_tokens)
        
        if current_chunk_tokens:
            chunks.append(self.tokenizer.decode(current_chunk_tokens))
            
        return [c.strip() for c in chunks if c.strip()]

    def _generate_nomic_embeddings_batch(self, texts: List[str], prefix: str, max_len: int) -> np.ndarray:
        """Generates Nomic embeddings for a batch of texts."""
        if not texts: return np.array([])
        
        prefixed_texts = [prefix + text for text in texts]
        encoded_input = self.tokenizer(
            prefixed_texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        return normalized_embeddings.cpu().numpy()

    def process_and_save_embeddings(self, json_path: str, output_path: str, batch_size: int = 32):
        """Reads articles, chunks them, creates embeddings, and saves to a file."""
        if not IJSON_AVAILABLE:
            print("FATAL: ijson library not found. Please install it to proceed.")
            return

        print(f"Starting to process articles from {json_path} using a streaming parser...")
        all_chunks_data = []
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                articles_iterator = ijson.kvitems(f, '')
                
                article_batch = []
                for article_id, article_data in tqdm(articles_iterator, desc="Processing Articles"):
                    article_batch.append((article_id, article_data))
                    
                    if len(article_batch) >= batch_size:
                        texts_to_chunk = []
                        article_ids_for_batch = []
                        
                        for art_id, art_data in article_batch:
                            if isinstance(art_data, dict) and 'content' in art_data:
                                title = art_data.get('title', '')
                                content = art_data.get('content', '')
                                full_text = f"{title}\n\n{content}"
                                texts_to_chunk.append(full_text)
                                article_ids_for_batch.append(art_id)

                        for idx, text in enumerate(texts_to_chunk):
                            article_id_for_chunk = article_ids_for_batch[idx]
                            
                            if self.text_splitter:
                                chunks = self.text_splitter.split_text(text)
                            else:
                                chunks = self._custom_text_splitter(text, CHUNK_SIZE, CHUNK_OVERLAP)

                            if not chunks: continue
                            
                            chunk_embeddings = self._generate_nomic_embeddings_batch(
                                chunks, "search_document: ", CHUNK_SIZE
                            )
                            
                            for chunk_text, chunk_embedding in zip(chunks, chunk_embeddings):
                                all_chunks_data.append({
                                    'article_id': article_id_for_chunk,
                                    'chunk_text': chunk_text,
                                    'embedding': chunk_embedding
                                })
                        
                        article_batch = [] # Reset batch

                # Process the last partial batch if any
                if article_batch:
                    texts_to_chunk = []
                    article_ids_for_batch = []
                    for art_id, art_data in article_batch:
                        if isinstance(art_data, dict) and 'content' in art_data:
                            title = art_data.get('title', '')
                            content = art_data.get('content', '')
                            full_text = f"{title}\n\n{content}"
                            texts_to_chunk.append(full_text)
                            article_ids_for_batch.append(art_id)

                    for idx, text in enumerate(texts_to_chunk):
                        article_id_for_chunk = article_ids_for_batch[idx]
                        if self.text_splitter:
                            chunks = self.text_splitter.split_text(text)
                        else:
                            chunks = self._custom_text_splitter(text, CHUNK_SIZE, CHUNK_OVERLAP)
                        if not chunks: continue
                        chunk_embeddings = self._generate_nomic_embeddings_batch(
                            chunks, "search_document: ", CHUNK_SIZE
                        )
                        for chunk_text, chunk_embedding in zip(chunks, chunk_embeddings):
                            all_chunks_data.append({
                                'article_id': article_id_for_chunk,
                                'chunk_text': chunk_text,
                                'embedding': chunk_embedding
                            })

        except ijson.JSONError as e:
            print(f"Error parsing JSON stream from {json_path}: {e}")
            return
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return

        print(f"Generated {len(all_chunks_data)} chunk embeddings. Saving to {output_path}...")
        with open(output_path, 'wb') as f:
            pickle.dump(all_chunks_data, f)
        print("Chunk embeddings saved.")

class BM25Indexer:
    """Handles creation of BM25 index for article text."""
    def __init__(self):
        self.bm25_model = None
        self.article_ids = []
        self.article_texts = []
        
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocesses text for BM25 indexing."""
        if not text:
            return []
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and split into words
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = text.split()
        
        # Filter out empty strings and very short words
        words = [word for word in words if len(word) > 1]
        
        return words
    
    def create_and_save_index(self, json_path: str, output_path: str):
        """Creates BM25 index from articles and saves it."""
        if not IJSON_AVAILABLE:
            print("FATAL: ijson library not found. Please install it to proceed.")
            return
            
        print(f"Creating BM25 index from {json_path}...")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                articles_iterator = ijson.kvitems(f, '')
                
                for article_id, article_data in tqdm(articles_iterator, desc="Processing Articles for BM25"):
                    if isinstance(article_data, dict) and 'content' in article_data:
                        title = article_data.get('title', '')
                        content = article_data.get('content', '')
                        full_text = f"{title}\n\n{content}"
                        
                        self.article_ids.append(article_id)
                        self.article_texts.append(full_text)
                        
        except ijson.JSONError as e:
            print(f"Error parsing JSON stream from {json_path}: {e}")
            return
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return
        
        print(f"Preprocessing {len(self.article_texts)} articles for BM25...")
        
        # Preprocess all articles
        processed_articles = [self._preprocess_text(text) for text in tqdm(self.article_texts, desc="Preprocessing text")]
        
        # Create BM25 index
        if BM25_AVAILABLE:
            self.bm25_model = BM25Okapi(processed_articles, k1=BM25_K1, b=BM25_B)
        else:
            # Fallback to simple BM25 implementation
            self.bm25_model = self._create_simple_bm25(processed_articles)
        
        # Save the index
        index_data = {
            'bm25_model': self.bm25_model,
            'article_ids': self.article_ids,
            'article_texts': self.article_texts
        }
        
        print(f"Saving BM25 index to {output_path}...")
        with open(output_path, 'wb') as f:
            pickle.dump(index_data, f)
        print("BM25 index saved.")
    
    def _create_simple_bm25(self, processed_articles):
        """Simple BM25 implementation as fallback."""
        # This is a simplified version - for production, use rank_bm25
        class SimpleBM25:
            def __init__(self, corpus, k1=1.2, b=0.75):
                self.k1 = k1
                self.b = b
                self.corpus = corpus
                self.doc_freqs = []
                self.idf = {}
                self.doc_len = []
                self.avgdl = 0
                
                # Calculate document frequencies and lengths
                for doc in corpus:
                    self.doc_len.append(len(doc))
                    
                self.avgdl = sum(self.doc_len) / len(self.doc_len)
                
                # Calculate IDF values
                df = defaultdict(int)
                for doc in corpus:
                    for word in set(doc):
                        df[word] += 1
                        
                for word, freq in df.items():
                    self.idf[word] = np.log((len(corpus) - freq + 0.5) / (freq + 0.5))
            
            def get_scores(self, query):
                scores = []
                for i, doc in enumerate(self.corpus):
                    score = 0
                    doc_len = self.doc_len[i]
                    for word in query:
                        if word in doc:
                            tf = doc.count(word)
                            idf = self.idf.get(word, 0)
                            score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
                    scores.append(score)
                return np.array(scores)
        
        return SimpleBM25(processed_articles, k1=BM25_K1, b=BM25_B)

# --- 4. ONLINE RETRIEVAL ---

class SearchPipeline:
    """Orchestrates the online search and re-ranking process."""
    def __init__(self, config: Dict, device: str = "cuda"):
        self.config = config
        self.device = device
        
        # Load models
        print("Loading models for online retrieval...")
        self.nomic_model = AutoModel.from_pretrained(config['TEXT_EMBEDDER_MODEL'], trust_remote_code=True).eval().to(device)
        self.nomic_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            config['IMAGE_ENCODER_MODEL'], pretrained=config['IMAGE_ENCODER_PRETRAINED'], device=device
        )
        self.clip_tokenizer = open_clip.get_tokenizer(config['IMAGE_ENCODER_MODEL'])
        print("All online models loaded.")

        # Load preprocessed data
        print("Loading pre-computed embeddings...")
        with open(config['IMAGE_EMBEDDINGS_PATH'], 'rb') as f:
            image_data = pickle.load(f)
            # Handle different file formats
            if 'ids' in image_data:
                self.image_ids = image_data['ids']
                self.image_embeddings = torch.tensor(image_data['embeddings']).to(self.device)
            else:
                # Handle legacy format
                self.image_ids = [os.path.splitext(os.path.basename(p))[0] for p in image_data['image_paths']]
                self.image_embeddings = torch.tensor(image_data['image_embeddings']).to(self.device)

        with open(config['CHUNK_EMBEDDINGS_PATH'], 'rb') as f:
            self.chunk_data = pickle.load(f)
            self.chunk_embeddings = torch.tensor(np.vstack([d['embedding'] for d in self.chunk_data])).to(self.device)

        with open(config['DATA_JSON_PATH'], 'r') as f:
            self.article_db = json.load(f)
            
        # Load BM25 index
        with open(config['BM25_INDEX_PATH'], 'rb') as f:
            bm25_data = pickle.load(f)
            self.bm25_model = bm25_data['bm25_model']
            self.bm25_article_ids = bm25_data['article_ids']
            
        print("Embeddings, BM25 index, and databases loaded.")
            
    def _embed_query_nomic(self, query: str) -> torch.Tensor:
        """Embeds a query using the Nomic model."""
        encoded_input = self.nomic_tokenizer(
            "search_query: " + query, padding=True, truncation=True, max_length=CHUNK_SIZE, return_tensors='pt'
        ).to(self.device)
        with torch.no_grad():
            model_output = self.nomic_model(**encoded_input)
        embedding = mean_pooling(model_output, encoded_input['attention_mask'])
        embedding = F.layer_norm(embedding, normalized_shape=(embedding.shape[1],))
        return F.normalize(embedding, p=2, dim=1)

    def _embed_query_clip(self, query: str) -> torch.Tensor:
        """Embeds a query using the CLIP text encoder."""
        with torch.no_grad():
            tokens = self.clip_tokenizer([query]).to(self.device)
            embedding = self.clip_model.encode_text(tokens)
            embedding /= embedding.norm(dim=-1, keepdim=True)
        return embedding
    
    def _preprocess_query_for_bm25(self, query: str) -> List[str]:
        """Preprocesses query for BM25 search (same as indexing preprocessing)."""
        if not query:
            return []
        
        # Convert to lowercase
        query = query.lower()
        
        # Remove punctuation and split into words
        query = query.translate(str.maketrans('', '', string.punctuation))
        words = query.split()
        
        # Filter out empty strings and very short words
        words = [word for word in words if len(word) > 1]
        
        return words
    
    def _reciprocal_rank_fusion(self, nomic_rankings: List[Tuple[str, float]], 
                               bm25_rankings: List[Tuple[str, float]], 
                               k_rrf: int = 60) -> List[Tuple[str, float]]:
        """Implements Reciprocal Rank Fusion to combine Nomic and BM25 rankings."""
        # Create dictionaries for easy lookup
        nomic_dict = {article_id: (rank + 1, score) for rank, (article_id, score) in enumerate(nomic_rankings)}
        bm25_dict = {article_id: (rank + 1, score) for rank, (article_id, score) in enumerate(bm25_rankings)}
        
        # Get all unique article IDs
        all_article_ids = set(nomic_dict.keys()) | set(bm25_dict.keys())
        
        # Calculate RRF scores
        rrf_scores = []
        for article_id in all_article_ids:
            rrf_score = 0
            
            # Add Nomic contribution
            if article_id in nomic_dict:
                nomic_rank = nomic_dict[article_id][0]
                rrf_score += 1 / (k_rrf + nomic_rank)
            
            # Add BM25 contribution
            if article_id in bm25_dict:
                bm25_rank = bm25_dict[article_id][0]
                rrf_score += 1 / (k_rrf + bm25_rank)
            
            rrf_scores.append((article_id, rrf_score))
        
        # Sort by RRF score descending
        rrf_scores.sort(key=lambda x: x[1], reverse=True)
        return rrf_scores

    def search(self, query: str) -> List[str]:
        """Executes the full hybrid retrieval and re-ranking pipeline."""
        # 1. Embed query for Nomic retrieval
        query_nomic_emb = self._embed_query_nomic(query)
        
        # 2. Preprocess query for BM25 retrieval
        query_bm25_tokens = self._preprocess_query_for_bm25(query)

        # 3A. Nomic Dense Retrieval for Articles
        chunk_sims = (query_nomic_emb @ self.chunk_embeddings.T).squeeze(0)
        top_m_indices = torch.topk(chunk_sims, min(self.config['TOP_M_CHUNKS'], len(chunk_sims)), largest=True).indices

        # Aggregate chunks to score articles (MaxP) and track best supporting chunks
        nomic_article_scores = {}
        article_best_chunk_scores = {}  # article_id -> best chunk's Nomic similarity to query
        
        for idx in top_m_indices:
            idx = idx.item()
            article_id = self.chunk_data[idx]['article_id']
            score = chunk_sims[idx].item()
            
            # Update article's max score
            if article_id not in nomic_article_scores or score > nomic_article_scores[article_id]:
                nomic_article_scores[article_id] = score
            
            # Track best chunk score per article (for C1 modulator)
            if article_id not in article_best_chunk_scores or score > article_best_chunk_scores[article_id]:
                article_best_chunk_scores[article_id] = score
        
        # Create Nomic ranking
        nomic_rankings = sorted(nomic_article_scores.items(), key=lambda item: item[1], reverse=True)
        
        # 3B. BM25 Sparse Retrieval for Articles
        bm25_article_scores = {}
        if query_bm25_tokens:  # Only if we have valid tokens
            bm25_scores = self.bm25_model.get_scores(query_bm25_tokens)
            for i, score in enumerate(bm25_scores):
                if i < len(self.bm25_article_ids):  # Safety check
                    article_id = self.bm25_article_ids[i]
                    bm25_article_scores[article_id] = float(score)
        
        # Create BM25 ranking
        bm25_rankings = sorted(bm25_article_scores.items(), key=lambda item: item[1], reverse=True)
        
        # 3C. Hybrid Fusion using Reciprocal Rank Fusion (RRF)
        hybrid_rankings = self._reciprocal_rank_fusion(
            nomic_rankings, bm25_rankings, self.config['K_RRF']
        )
        
        # 3D. Select Top-N Articles based on hybrid scores
        top_n_articles = hybrid_rankings[:self.config['TOP_N_ARTICLES']]
        
        if not top_n_articles:
            return ['#'] * self.config['TOP_K_IMAGES']

        # Normalize article scores for re-ranking (use original Nomic scores for C1)
        max_nomic_score = max(nomic_article_scores.values()) if nomic_article_scores else 1.0
        normalized_article_scores = {}
        for article_id, _ in top_n_articles:
            if article_id in nomic_article_scores:
                normalized_article_scores[article_id] = nomic_article_scores[article_id] / max_nomic_score
            else:
                normalized_article_scores[article_id] = 0.0
        
        # 5. Collect candidate images from top articles
        candidate_image_ids = set()
        for article_id, _ in top_n_articles:
            images_in_article = self.article_db.get(article_id, {}).get('images', [])
            candidate_image_ids.update(images_in_article)
        
        candidate_image_ids = list(candidate_image_ids)
        if not candidate_image_ids:
            return ['#'] * self.config['TOP_K_IMAGES']

        # 6. Embed query for holistic visual comparison
        query_clip_emb = self._embed_query_clip(query)

        # 7. Re-rank images with chunk similarity multiplier
        final_image_scores = {}
        for image_id in candidate_image_ids:
            try:
                # Find the image's pre-computed embedding
                idx = self.image_ids.index(image_id)
                image_emb = self.image_embeddings[idx]
                
                # Find parent article
                parent_article_id = next((art_id for art_id, data in self.article_db.items() if image_id in data.get('images', [])), None)
                if parent_article_id is None:
                    continue
                
                # Get normalized article score
                norm_article_score = normalized_article_scores.get(parent_article_id, 0)
                
                # Calculate holistic image similarity score (query vs image)
                holistic_image_sim_score = (query_clip_emb @ image_emb.T).item()
                
                # Apply chunk similarity multiplier if enabled
                if self.config['USE_CHUNK_MULTIPLIER'] and parent_article_id in article_best_chunk_scores:
                    # Get the best chunk's Nomic similarity to query for this article
                    nomic_best_chunk_to_query_sim = article_best_chunk_scores[parent_article_id]
                    
                    # Apply multiplier: holistic_score * (1 + lambda * chunk_sim)
                    multiplier = 1 + self.config['LAMBDA_CHUNK_BOOST'] * nomic_best_chunk_to_query_sim
                    boosted_visual_score = holistic_image_sim_score * multiplier
                else:
                    boosted_visual_score = holistic_image_sim_score
                
                # Final combined score using multiplier approach
                final_score = (
                    self.config['W1_HOLISTIC_VISUAL'] * boosted_visual_score +
                    self.config['W3_ARTICLE_SCORE'] * norm_article_score
                )
                
                final_image_scores[image_id] = final_score

            except (ValueError, IndexError):
                continue # Image not found in our pre-computed embeddings
        
        # 8. Sort and return top K image IDs
        sorted_images = sorted(final_image_scores.items(), key=lambda item: item[1], reverse=True)
        top_image_ids = [img_id for img_id, score in sorted_images[:self.config['TOP_K_IMAGES']]]
        
        # Pad if necessary
        while len(top_image_ids) < self.config['TOP_K_IMAGES']:
            top_image_ids.append('#')
            
        return top_image_ids

# --- 5. MAIN EXECUTION ---

def run_offline_processing():
    """Checks for embedding files and generates them if they don't exist."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Process Images
    if not os.path.exists(IMAGE_EMBEDDINGS_PATH):
        print("--- Running Offline Image Processing ---")
        embedder = ImageEmbedder(IMAGE_ENCODER_MODEL, IMAGE_ENCODER_PRETRAINED, device)
        embedder.create_and_save_embeddings(IMAGE_FOLDER_PATH, IMAGE_EMBEDDINGS_PATH)
        del embedder
        gc.collect()
    else:
        print(f"Found existing image embeddings: {IMAGE_EMBEDDINGS_PATH}")

    # Process Articles
    if not os.path.exists(CHUNK_EMBEDDINGS_PATH):
        print("--- Running Offline Article Processing ---")
        embedder = ArticleEmbedder(TEXT_EMBEDDER_MODEL, device)
        embedder.process_and_save_embeddings(DATA_JSON_PATH, CHUNK_EMBEDDINGS_PATH)
        del embedder
        gc.collect()
    else:
        print(f"Found existing chunk embeddings: {CHUNK_EMBEDDINGS_PATH}")
        
    # Process BM25 Index
    if not os.path.exists(BM25_INDEX_PATH):
        print("--- Running Offline BM25 Index Creation ---")
        indexer = BM25Indexer()
        indexer.create_and_save_index(DATA_JSON_PATH, BM25_INDEX_PATH)
        del indexer
        gc.collect()
    else:
        print(f"Found existing BM25 index: {BM25_INDEX_PATH}")

def process_queries_and_create_submission(query_file_path: str, output_csv_path: str):
    """Processes a query file and generates the final submission CSV."""
    start_time = time.time()
    
    # Prepare pipeline configuration from globals
    config = {
        'TEXT_EMBEDDER_MODEL': TEXT_EMBEDDER_MODEL,
        'IMAGE_ENCODER_MODEL': IMAGE_ENCODER_MODEL,
        'IMAGE_ENCODER_PRETRAINED': IMAGE_ENCODER_PRETRAINED,
        'IMAGE_EMBEDDINGS_PATH': IMAGE_EMBEDDINGS_PATH,
        'CHUNK_EMBEDDINGS_PATH': CHUNK_EMBEDDINGS_PATH,
        'BM25_INDEX_PATH': BM25_INDEX_PATH,
        'DATA_JSON_PATH': DATA_JSON_PATH,
        'TOP_M_CHUNKS': TOP_M_CHUNKS,
        'TOP_N_ARTICLES': TOP_N_ARTICLES,
        'TOP_K_IMAGES': TOP_K_IMAGES,
        'W1_HOLISTIC_VISUAL': W1_HOLISTIC_VISUAL,
        'W3_ARTICLE_SCORE': W3_ARTICLE_SCORE,
        'LAMBDA_CHUNK_BOOST': LAMBDA_CHUNK_BOOST,
        'USE_CHUNK_MULTIPLIER': USE_CHUNK_MULTIPLIER,
        'K_RRF': K_RRF
    }
    
    # Initialize the search pipeline
    try:
        pipeline = SearchPipeline(config)
    except Exception as e:
        print(f"FATAL: Could not initialize search pipeline: {e}")
        return

    # Read queries
    try:
        df_queries = pd.read_csv(query_file_path)
    except Exception as e:
        print(f"Error reading query file {query_file_path}: {e}")
        return
        
    print(f"Found {len(df_queries)} queries to process.")
    
    # Process each query
    submission_results = []
    for _, row in tqdm(df_queries.iterrows(), total=len(df_queries), desc="Processing Queries"):
        query_id = row['query_index']
        query_text = row['query_text']
        
        if pd.isna(query_text) or not query_text.strip():
            print(f"Warning: Empty query text for {query_id}. Skipping.")
            top_images = ['#'] * TOP_K_IMAGES
        else:
            top_images = pipeline.search(query_text)
            
        submission_results.append([query_id] + top_images)

    # Write submission file
    print(f"Writing submission file to: {output_csv_path}")
    header = ['query_id'] + [f'image_id_{i}' for i in range(1, TOP_K_IMAGES + 1)]
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(submission_results)

    total_time = time.time() - start_time
    print("\n--- Processing Complete ---")
    print(f"Submission file created successfully with {len(submission_results)} rows.")
    print(f"Total time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes).")

if __name__ == "__main__":
    # Step 1: Ensure all data is preprocessed and embeddings are created.
    run_offline_processing()
    
    # Step 2: Run the query processing and generate the final submission file.
    process_queries_and_create_submission(QUERY_FILE_PATH, SUBMISSION_CSV_PATH) 