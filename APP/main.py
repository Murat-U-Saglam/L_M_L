import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf

from PIL import Image


def main():
    st.title("Cifar10 Classification")
    st.write("Upload an image, let me classify it for you")
    file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="image")
    if file:
        image = Image.open(file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        resized_image = image.resize((32,32))
        image_array = np.array(resized_image)
        image_array = image_array / 255
        image_array = image_array.reshape(1,32,32,3)


        model = tf.keras.models.load_model('model.h5')
        prediction = model.predict(image_array)
        cifar10_labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
        st.write("Classifying...")
        fig, ax = plt.subplots()
        y_pos = np.arange(len(cifar10_labels))
        ax.barh(y_pos, prediction[0], align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(cifar10_labels)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Probability')
        ax.set_title('Cifar10 Classification')
        st.pyplot(fig)
    else:
        st.text("You have not uploaded an image yet")

if __name__ == "__main__":
    main()
    