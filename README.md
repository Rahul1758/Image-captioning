# Image-captioning

1. DESCRIPTION
--------------
Image captioning is one of the most popular application of Encoder-Decoder architecture where CNN extracts and encodes the features of an image while RNN is used as Decoder.

2. DATA
--------------
The dataset for this project is Flickr_8k dataset. It contains 8000 images along with text file containing 5 captions for each image. 

3. MODEL TRAINING
--------------
The libraries used in this project are Tensorflow, NLTK, pandas & numpy.
To encode the feature images Inception_V3 model was used with transfer learning to download the weights. Decoder has Bidirectional LSTM followed by an LSTM layer in it. The approach taken for this project is Merge model wherein the Encoder separately extracts the features from the image and RNN separately trains on the captions after which these information is concatenated and passed through Fully-connected layers to get final prediction.

4. DEPLOYMENT
--------------
The app uses Streamlit framework to create a front-end API and is deployed on Heroku.
Streamlit is framework using which you can create aesthetic front-end Webapp without the need of HTML and CSS. You can check their website here. https://www.streamlit.io/. 
Please feel free to try out the app by hitting this link!! https://image-captioning-12.herokuapp.com/

5. INSIGHTS
--------------
- Model can be trained on much larger datasets to achieve more accurate caption generation on the test image, reason being if only the model has been trained on an image similar to the test image then it has the chance to be provide good predictions. For that to happen the model needs to be trained on huge datasets.
- Bleu score for beam width=5 was used as metric due to computation time & resource constraint. Try for beam width>5 which would give better Bleu score.
- The model was trained for 30 epochs. Try training for more epochs for better results but also avoiding overfitting.
- Implementing caption generation with attention architecture will also improve the model.
