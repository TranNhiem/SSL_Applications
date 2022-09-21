##TranNhiem 2022/07 
'''
Sampling prompts with the input prompt using LLM Pretrained model 

1. Using GPT2 model 
2. Using OPT Model 
3. Using BLOOM Model 

'''
import torch 
##--------------------------------------------------------
##1. Using GPT2 model Better using Distill model 
##--------------------------------------------------------
from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)

##--------------------------------------------------------
##2. Using OPT model Better using Distill model 
##--------------------------------------------------------
from transformers import OPTModel, OPTConfig


##--------------------------------------------------------
##3. Using OPT model Better using Distill model 
##--------------------------------------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
torch.set_default_tensor_type(torch.cuda.FloatTensor)
### Loading Pretrained weight and Tokenizer 
bloom_model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-1b3", use_cache=True)
bloom_tokenizer= AutoTokenizer.from_pretrained("bigscience/bloom-1b3")
### Reproducible behavior to set the seed in random, numpy, torch a
set_seed(1234)
prompt="An image of hummingbirds flying over Tulip flowers"
input= bloom_tokenizer(prompt, return_tensors="pt").to(0)
## Using TopK output 
output_prompts = bloom_model.generate(**input_ids, max_length=200,  top_k=1, temperature=0.9, repetition_penalty = 2.0)

## Using Beam Search to sample the Outputs 
output_prompts = bloom_model.generate(**input_ids, max_length=200, num_beams = 2, num_beam_groups = 2, top_k=1, temperature=0.9, repetition_penalty = 2.0)


output_prompts=bloom_tokenizer.decode(output_prompts[0],  truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"])
