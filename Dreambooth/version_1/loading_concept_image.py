from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class DreamBoothDataset(Dataset): 
    def __init__(self, root_dir,instance_prompt, tokenizer, class_prompt, size,class_data_root=None,  center_crop=False, transform=None):
        '''
        return the image and it instance_prompt (Concept prompt)
        '''
        
        self.root_dir = Path(root_dir)
        #self.image_paths = list(self.root_dir.glob('*.jpg'))
        if not self.root_dir.exists(): 
            raise ValueError(f"Root directory {self.root_dir} does not exist")

        self.instance_images_path = list(Path(root_dir).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self._length= self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length= max(self.num_instance_images,self.num_class_images)
            self.class_prompt= class_prompt
        else: 
            self.class_data_root= None 
        
        if transform is not None: 

            self.image_transforms = transform
        else:
            self.image_transforms= transforms.Compose([
                transforms.Resize(size,interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                transforms.Normalize([0.5], [0.5])

            ])
            

        self.zie= size,
        self.center_crop=center_crop
        self.class_prompt=class_prompt
        self.tokenizer=tokenizer
        self.instance_prompt=instance_prompt
       
       

        
    def __len__(self):
        return self._length
    
    # def __getitem__(self, idx):
    #     example={}
    #     instance_image= Image.open(self.instance_images_path[idx % self.num_instance_images]).convert('RGB')
    #     if not instance_image.mode == "RGB":
    #         instance_image = instance_image.convert("RGB")
    #     example["instance_images"] = self.image_transforms(instance_image)
    #     example["instance_prompt_ids"] = self.tokenizer(self.instance_prompt,padding="do_not_pad", 
    #                                             truncation=True, max_length=self.tokenizer.model_max_length)

    #     if self.class_data_root is not None:
    #         class_image= Image.open(self.class_images_path[idx% self.num_class_images])
    #         if not class_image.mode == "RGB":
    #             class_image = class_image.convert("RGB")
    #         example["class_images"] = self.image_transforms(class_image)
    #         example["class_prompt_ids"]= self.tokenizer(self.class_prompt, 
    #                                             padding="do_not_pad",
    #                                             truncation=True, max_length=self.tokenizer.model_max_length)

    #     return example
    def __getitem__(self, index):
        example = {}
        # print(self.instance_images_path[index % self.num_instance_images])
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids
        
        return example
## Create the prompt dataset
class PromptDataset(Dataset): 
    
    def __init__(self, prompt, num_samples): 
        self.prompt= prompt
        self.num_samples= num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        example={}
        example["prompt"]= self.prompt
        example["index"]= idx 
        return example

