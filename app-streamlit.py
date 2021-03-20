from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, model_from_json
from PIL import Image
import numpy as np
import pandas as pd
import pickle 
import streamlit as st
from tempfile import NamedTemporaryFile

max_length=34
tokenizer=pickle.load(open('train_tokenizer.pkl','rb'))
json_file = open('model_30.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
caption_model = model_from_json(loaded_model_json)
caption_model.load_weights("model_30_wts.h5")

def main():
    html_temp = """
        <div style="background-color:tomato;padding:10px">
       <h2 style="color:white;text-align:center;">Image captioning app</h2>
       </div>
       """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.write('''This app has been made using STREAMLIT framework to predict the suitable caption for your input image. Let's Start!''')
    st.header('Please choose an image for Prediction')
    file=st.file_uploader('Choose image',type=['png','jpeg','jpg'])
    temp_file = NamedTemporaryFile(delete=False)
    if file is not None:
        img=Image.open(file)
        st.image(img,use_column_width=True)
        temp_file.write(file.getvalue())
        photo_feature=extract_photo_features(temp_file.name)
        caption,scores=Beam_search(10,tokenizer,max_length, caption_model, photo_feature)
        table=pd.DataFrame(caption, columns=['captions'])
        table['scores']=np.round(scores,decimals=4)
        table=table.sort_values('scores',ascending=False)
        table.drop_duplicates('captions',inplace=True)
        table=table.reset_index(drop=True)
        result=(table['captions'][0]).capitalize()
        st.write('Photo caption: {}'.format(result))
              

@st.cache()
def extract_photo_features(filename):
    inception=InceptionV3(weights='imagenet')
    for layer in inception.layers:
        layer.trainable=False
    model=Model(inputs=inception.input, outputs=inception.layers[-2].output)
    image=load_img(filename,target_size=(299,299))
    image=img_to_array(image)
    image=np.expand_dims(image,axis=0)
    image=preprocess_input(image)
    feature=model.predict(image,verbose=0)
    return feature

@st.cache()
def Beam_search(beam_width, tokenizer, max_length, model, photo_feature):
    seq=tokenizer.texts_to_sequences(['<START>'])[0]
    in_text=[[seq,0.0]]

    while len(in_text[0][0])<max_length:
        temp=list()
        for i in in_text:
            seq=pad_sequences([i[0]], maxlen=max_length)
            output_softmax=model.predict([photo_feature, seq])
            most_likely_seq=np.argsort(output_softmax[0])[-beam_width:]

        for j in most_likely_seq:
            next_seq, prob= i[0][:], i[1]
            next_seq.append(j)
            prob = (prob+ np.log(output_softmax[0][j]))/len(next_seq)
            temp.append([next_seq,prob])
      
        in_text=temp
        in_text= sorted(in_text, key= lambda x:x[1])
        in_text=in_text[-beam_width:]
  
    most_likely_idx=in_text
    scores= [i[1] for i in most_likely_idx]
    most_likely_cap=[[tokenizer.index_word[idx] for idx in j[0]]for j in most_likely_idx]

    best_caption=list()
    for i in range(len(most_likely_cap)):
        best_caption.append(list())
        for k in most_likely_cap[i]:
            if k!='end':
                best_caption[i].append(k)
            else:
                break
        best_caption[i]=' '.join(best_caption[i][1:])
    return best_caption,scores


if __name__=='__main__':
    main()