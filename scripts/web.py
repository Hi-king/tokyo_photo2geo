import streamlit as st
import torch
import json
import pathlib
import photo2geo
import PIL.Image
import torchvision
import numpy as np
import helper

RESULT_PATH = (pathlib.Path(__file__).parent.parent / "results/class7_resnet50_batch30_lr1e-05_commit6241653_202112070034").as_posix()


def myapp():
    st.write("hello")

    file_up = st.file_uploader("Upload an image", type=['png','jpg'])

    if file_up is not None:
        model, params = helper.load_params(RESULT_PATH)
        image = PIL.Image.open(file_up)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        predicted, scores = helper.pred(file_up, model, params)
        st.text(f'予測: {predicted}')

        TOPK=3
        cols = st.columns(TOPK)
        for col, (key, score) in zip(cols, list(sorted(scores.items(), key=lambda x: -x[1]))[:TOPK]):
            with col:
                cam_image = helper.show_cam(file_up, model, params, key)
                st.image(cam_image, caption=f'GradCAM {key} [{score:.2f}]', use_column_width=True)

if __name__ == "__main__":
    myapp()
