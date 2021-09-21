import numpy as np
import tensorflow as tf
keras = tf.keras
K = keras.backend
import matplotlib.pyplot as plt
import pandas as pd


class Word2id():
    def __init__(self):
        self.word2id = dict()
        self.id2word = dict()
    
    def fit(self, text):
        T = sorted(set(text))
        for t in T:
            if t not in self.word2id.keys():
                self.word2id[t] = len(self.word2id)
                self.id2word[self.word2id[t]] = t
    
    def transform(self, text):
        return [self.word2id[t] for t in text]
    
    def fit_transform(self, text):
        self.fit(text)
        return self.transform(text)
    
    def transform_id2word(self, ids):
        text = [self.id2word[id_] for id_ in ids]
        return text
    
    def get_vocabulary_words(self):
        v=self.word2id.values()
        words = [self.id2word[vv] for vv in range(min(v), max(v)+1)]
        return words

class HistoryLoss(keras.callbacks.Callback):
    def __init__(self):
        self.ls = list()
    
    def on_train_batch_end(self, batch, logs=None):
        self.ls.append(logs["loss"])
        
    def show(self):
        plt.plot(self.ls)
        plt.title("loss per batch")
        plt.grid()
        plt.xlabel("batch")
        plt.show()

class DeepLogModelCore():
    def __init__(self, h, n, vocabulary):
        self.h = h
        self.n = n
        self.W2ID = Word2id()
        self.W2ID.fit(vocabulary)
        
    def build(self, num_lstm_layers, lstm_size, learning_rate=0.01):
        h = self.h
        n = self.n
        
        input_layer = keras.layers.Input(shape=(h, n), dtype='float32', name="input")
        
        last_layer = input_layer
        for lstm_layer_index in range(1,num_lstm_layers):
            last_layer = keras.layers.LSTM(lstm_size, return_sequences=True, name="lstm-{}".format(lstm_layer_index))(last_layer)
        lstm_layer = keras.layers.LSTM(lstm_size, return_sequences=False, name="lstm-{}".format(num_lstm_layers))(last_layer)
        forward = keras.layers.Dense(n, activation="softmax", name="forward")(lstm_layer)
        
        model = keras.models.Model(inputs=[input_layer], outputs=[forward])
        loss_fn = keras.losses.categorical_crossentropy
        optimizer = keras.optimizers.RMSprop(lr=learning_rate)
        model.compile(loss=loss_fn, optimizer=optimizer)
        self.model = model

    def fit(self, text, epochs=1):
        X, y, y_ids = self.Xy(text)
        h = HistoryLoss()
        self.model.fit(X,y,batch_size=128, epochs=epochs,verbose=0, callbacks=[h])
        return h
    
    def predict_vectors(self, text):
        X, y, y_ids = self.Xy(text)
        return self.model.predict(X)
     
    def Xy(self, text):
        ids = self.W2ID.fit_transform(text)
        E = np.eye(self.n)
        one_hots = [E[id_].copy().tolist() for id_ in ids]
        
        h = self.h
        X = np.array    ([one_hots[i :  i+h] for i in range(len(one_hots)-h)])
        y = np.array    ([one_hots[     i+h] for i in range(len(one_hots)-h)])
        y_ids = np.array([ids     [     i+h] for i in range(len(ids)     -h)])
        
        assert X.shape == (len(one_hots)-h, h, self.n)
        assert y.shape == (len(one_hots)-h, self.n)
        assert y_ids.shape == (len(one_hots)-h,)
        return X, y, y_ids

class DeepLogModel(DeepLogModelCore):
    def __init__(self, h, n, vocabulary):
        super().__init__(h, n, vocabulary)
        
    def get_df_pred(self, text, marks):
        X,y,y_ids = self.Xy(text)
        pred_vectors = self.predict_vectors(text)
        df_pred = pd.DataFrame(pred_vectors)
        df_pred.columns = self.W2ID.get_vocabulary_words()
        df_pred.index = self.W2ID.transform_id2word(y_ids)
        
        marks = marks[-len(y_ids):]
        df_marks = pd.DataFrame([pd.Series(self.W2ID.transform_id2word(y_ids), name = "letter"), pd.Series(marks, name = "mark")]).T
        
        return df_pred, df_marks
    
    def monitor_session(self, text, marks, g):
        anomaly_exists = 0 in marks
        
        report = list()
        df_pred, df_marks = self.get_df_pred(text, marks)
        for i, (letter, vector) in enumerate(df_pred.iterrows()):
            if g == 0:
                alert = True
            else:
                topg = vector.sort_values()[-g:].index
                ok = letter in topg
                alert = not ok
            report.append(alert)
        
        anomaly_found = True in report
        
        # D (anomaly_exists, anomaly_found) : classification
        D = { (False,False) : "TN",
              (False, True) : "FP",
              (True , False): "FN",
              (True , True) : "TP" }
        
        result = D[(anomaly_exists, anomaly_found)]
        return result
    
    def train_feedback(self, text, marks, g):
        marks = [1]*len(marks) # no zeros
        result = self.monitor_session(text, marks,g)
        while(result != "TN"):
            self.fit(text,epochs=1)
            result = self.monitor_session(text, marks,g)
        
        assert result == "TN"

    def monitor_session_extended(self, text, marks, expected_alert, g):
        X,y,y_ids = self.Xy(text)
        pred_vectors = self.predict_vectors(text)
        df_pred = pd.DataFrame(pred_vectors)
        df_pred.columns = self.W2ID.get_vocabulary_words()
        df_pred.index = self.W2ID.transform_id2word(y_ids)
        
        marks = marks[-len(y_ids):]
        df_marks = pd.DataFrame([pd.Series(self.W2ID.transform_id2word(y_ids), name = "letter"), pd.Series(marks, name = "mark")]).T
        
        entries = list()
        for i, (letter, vector) in enumerate(df_pred.iterrows()):
            topg = vector.sort_values()[-g:].index
            ok = letter in topg
            alert = not ok
            entry = {"i" : i,
                     "letter" : letter,
                     "p" : vector[letter],
                     "alert" : alert}
            entries.append(entry)
        
        df_monitor = pd.DataFrame(entries)
        df_monitor["mark"] = df_marks["mark"]
        found_alert = True in list(df_monitor["alert"])
        
        key = (expected_alert, found_alert)
        res = {(False, False) : "TN",
         (False, True) : "FP",
         (True, False) : "FN",
         (True, True) : "TP"}[key]
        
        return df_monitor, res
    
