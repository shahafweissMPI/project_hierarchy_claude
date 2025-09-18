import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from tqdm import tqdm

class GLM:
    def __init__(self, X, y, dt):
        # Convert inputs to NumPy arrays and preprocess.
        X = np.asarray(X)
        y = np.asarray(y)
        X = np.squeeze(X)
        y = np.squeeze(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self.dt = dt
        self.X = tf.constant(X, dtype=tf.float32)
        self.y = tf.constant(y, dtype=tf.float32)
        self.w = tf.Variable(tf.zeros([self.X.shape[1], 1], dtype=tf.float32))
    
    @tf.function
    def _neg_log_likelihood_batch(self, X_batch, y_batch):
        # Compute the Poisson log-likelihood in a batched manner.
        rate = tf.linalg.matmul(X_batch, self.w)
        rate = tf.exp(rate)
        eps = 1e-8  # For numerical stability.
        log_likelihood = y_batch * tf.math.log(rate + eps) - rate * self.dt
        return -tf.reduce_sum(log_likelihood)
    
    def neg_log_likelihood(self, batch_size=65536):
        if batch_size is None:
            return self._neg_log_likelihood_batch(self.X, self.y)
        
        dataset = tf.data.Dataset.from_tensor_slices((self.X, self.y)).batch(batch_size)
        total_loss = dataset.reduce(
            tf.constant(0.0, dtype=tf.float32),
            lambda state, batch: state + self._neg_log_likelihood_batch(batch[0], batch[1])
        )
        return total_loss

    def fit(self, learning_rate=0.01, epochs=1000, batch_size=65536):
        optimizer = tf.optimizers.Adam(learning_rate)
        dataset = tf.data.Dataset.from_tensor_slices((self.X, self.y))
        buffer_size = tf.cast(tf.shape(self.X)[0], tf.int64)
        dataset = dataset.shuffle(buffer_size=buffer_size).batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        @tf.function
        def train_step(x_batch, y_batch):
            with tf.GradientTape() as tape:
                loss = self._neg_log_likelihood_batch(x_batch, y_batch)
            grads = tape.gradient(loss, [self.w])
            optimizer.apply_gradients(zip(grads, [self.w]))
            return loss

        # Force training on GPU if available.
        with tf.device('/GPU:0'):
            for epoch in range(epochs):
                for x_batch, y_batch in dataset:
                    train_loss = train_step(x_batch, y_batch)
                # Optionally log the training loss.
                # tf.print("Epoch", epoch, "Loss", train_loss)

    def predict_rate(self, batch_size=65536):
        if batch_size is None:
            rate = tf.exp(tf.linalg.matmul(self.X, self.w))
            return (rate / self.dt).numpy()
        
        rates = []
        dataset = tf.data.Dataset.from_tensor_slices(self.X).batch(batch_size)
        for x_batch in dataset:
            rate_batch = tf.exp(tf.linalg.matmul(x_batch, self.w))
            rates.append(rate_batch)
        rate = tf.concat(rates, axis=0)
        return (rate / self.dt).numpy()

    @staticmethod
    def combined_feature_selection(X, y, predictor_names, dt=0.02, cv_folds=10, batch_size=65536, n_jobs=-1):
        """
        Combines forward selection with caching, a progress bar (tqdm), and
        parallel cross-validation using joblib.
        
        Args:
            X (np.ndarray): The design matrix.
            y (np.ndarray): The response variable.
            predictor_names (list): Names of candidate predictors.
            dt (float): Time bin size.
            cv_folds (int): Number of cross-validation folds.
            batch_size (int): Batch size for training/evaluation.
            n_jobs (int): Number of parallel jobs for cross-validation.
        
        Returns:
            best_subset_names (list): The names of the selected predictors.
            best_overall_performance (float): The associated avg negative log-likelihood.
        """
        n_features = X.shape[1]
        selected_indices = []
        remaining_indices = list(range(n_features))
        best_overall_performance = np.inf
        best_subset_names = []
        cache = {}  # Cache for evaluated feature subsets

        def evaluate_subset(feature_indices):
            key = tuple(sorted(feature_indices))
            if key in cache:
                return cache[key]
            subset_X = X[:, feature_indices]
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            folds = list(kf.split(subset_X))
            
            # Evaluate one fold: train on training indices and score on test indices.
            def evaluate_fold(train_index, test_index):
                X_train, X_test = subset_X[train_index], subset_X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                if y_train.ndim == 2 and y_train.shape[1] == 1:
                    y_train = y_train.ravel()
                if y_test.ndim == 2 and y_test.shape[1] == 1:
                    y_test = y_test.ravel()
                glm = GLM(X_train, y_train, dt=dt)
                glm.fit(batch_size=batch_size)
                test_glm = GLM(X_test, y_test, dt=dt)
                test_glm.w.assign(glm.w)
                loss = test_glm.neg_log_likelihood(batch_size=batch_size).numpy()
                return loss

            # Evaluate folds in parallel.
            performances = Parallel(n_jobs=n_jobs)(
                delayed(evaluate_fold)(train_index, test_index)
                for train_index, test_index in folds
            )
            avg_performance = np.mean(performances)
            cache[key] = avg_performance
            return avg_performance

        pbar = tqdm(total=len(remaining_indices), desc="Selecting features")
        improved = True
        while improved and remaining_indices:
            improved = False
            candidate_results = {}
            for i in remaining_indices:
                candidate_indices = selected_indices + [i]
                subset_name = [predictor_names[idx] for idx in candidate_indices]
                tqdm.write(f"Evaluating subset: {subset_name}")
                candidate_results[i] = evaluate_subset(candidate_indices)
            
            best_candidate = min(candidate_results, key=candidate_results.get)
            best_candidate_perf = candidate_results[best_candidate]

            if best_candidate_perf < best_overall_performance:
                best_overall_performance = best_candidate_perf
                selected_indices.append(best_candidate)
                best_subset_names.append(predictor_names[best_candidate])
                remaining_indices.remove(best_candidate)
                improved = True
                tqdm.write(f"--> Selected subset: {[predictor_names[idx] for idx in selected_indices]} with performance: {best_candidate_perf}")
            else:
                tqdm.write("No further improvement found.")
                break
            pbar.update(1)
        pbar.close()
        return best_subset_names, best_overall_performance