from datasets import load_dataset

dataset = load_dataset("imagefolder", data_dir="instance_images")
dataset.push_to_hub("ArchitKohli/vector-lora-dataset")
print(dataset["train"]._data)