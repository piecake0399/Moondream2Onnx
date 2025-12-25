# Moondream2Onnx
ONNX deployment for Moondream2

1. Install the dependencies: ``` pip install -r requirements.txt ```

2. Download the ONNX weight files from the following link: [https://huggingface.co/onnx-community/Florence-2-base-ft/tree/main/onnx](https://huggingface.co/Xenova/moondream2/tree/main/onnx)

   2.1. Copy the weight files to the weight_files folder (3 weight files are needed. Vision Encoder, Embed Tokens and Decoder Model Merged).

   For example: vision_encoder_fp16.onnx, embed_tokens_fp16.onnx, decoder_model_merged_q4.onnx

3. Run the following command ``` python md2.py ``` 

---------
