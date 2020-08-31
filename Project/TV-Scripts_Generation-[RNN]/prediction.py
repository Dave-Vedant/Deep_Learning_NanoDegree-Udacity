import torch
import torch.nn as nn
import torch.nn.functional as F

import helper
import project_unitTest as test
import generate_text 


# model prediction control parameters
gen_length = 400
prime_word = 'jerry'

pad_word = helper.special_words['PADDING']
generate_script = generate(train_rnn, vocab_to_int[prime_word + ':'], int_to_vocab,
        token_dict, vocab_to_int[pad_word], gen_length)
        
    
print(generated_script)

# save the script
f = open("generated_script.txt","w")
f.write(generate_script)
f.close()
