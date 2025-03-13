This is the README for Phase 2 of CPSC8470 Term Project

To be able to run the code first embeddings must be made. For either Flickr30k or MS COCO that means downloading the mdodel. I downloaded the model locally using:

wget -P models/clip-vit-large-patch14 https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/config.json
wget -P models/clip-vit-large-patch14 https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/pytorch_model.bin
wget -P models/clip-vit-large-patch14 https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/preprocessor_config.json
wget -P models/clip-vit-large-patch14 https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/tokenizer.json
wget -P models/clip-vit-large-patch14 https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/tokenizer_config.json

For the Flickr30k dataset to embeddings were completed running flicker30k_embedding.py. To run ms_coco_embeddings.py the 118k images were actually downloaded locally using:

wget http://images.cocodataset.org/zips/train2017.zip

once embeddings were completed the python scripts ann.py and coco_ann.py (changing the passed in parameters as appropriate) were run to generate the results.
