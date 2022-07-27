import streamlit as st
from imageai.Prediction import ImagePrediction
from PIL import Image

@st.cache
def load_image(img):
    im = Image.open(img)
    return im

@st.cache(allow_output_mutation=True)
def load_models(model_file):
    prediction = ImagePrediction()
    prediction.setModelTypeAsResNet()
    prediction.setModelPath(model_file)
    prediction.loadModel()
    return prediction

prediction = load_models("./models/resnet50_weights_tf_dim_ordering_tf_kernels.h5")

def main():
    st.title("Object Detection App")
    image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
    if image_file is not None:
        our_image = Image.open(image_file)
        st.text("Original Image")
        st.image(our_image,width=300)
        predictions, probabilities = prediction.predictImage(our_image,result_count=10,input_type='array')

        for eachPrediction, eachProbability in zip(predictions, probabilities):
            st.write(eachPrediction, " : ", eachProbability)

if __name__ == '__main__':
    main()
