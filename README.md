# Node-Neuron Network
A training platform for node-neuron neural network cascade concept for Mat-mul reduced transformer

Based on my transformer Trainer for Matmul-free or limited implementation

Hello! This is a training (and inference) program BASED on the Matmul-Free architecture outlined in 
Scalable MatMul-free Language Modeling
by Rui-Jie Zhu1
, Yu Zhang2
, Ethan Sifferman1
, Tyler Sheaves3
, Yiqiao Wang4
, Dustin Richmond1
, Peng Zhou1,4
, Jason K. Eshraghian1∗
1University of California, Santa Cruz 2Soochow University
3University of California, Davis 4LuxiTech

This is a basic implementation that still uses matrix multiplication on the linear pass, but it set up for me to upgrade it as I work through the difficulties combining this with the mat-mul free linear pass or the Bitnet bitlinear pass. It also uses matrix multiplication for self-attention and cross-node and global-node attention. 

This is based on the idea of reducing the necessary parameters and size of a transformer by using multiple embedding stages through separate nodes,  before the final token prediction layer. I have included a basic transformer model as well if you would prefer to use a standard transformer implementation, it seems to work just as well but I have it set up so that it can be easily adapted to remove matrix multiplications in the future to drastically increase its viability for putting massive Ais on consumer grade products.

Essentially, my thinking and principle behind this was to develop a neural network that functioned more similar to the human brain, a chemical transformer that performs parallel and sequential activation with a general principle of 40-50 percentof a single neuron's activation to be applied locally and globally respectively. To achieve this, I realized a different implementation would be necessary where instead of treating an embedding parameter as a neuron, each neuron would be an individual transformer or gated unit and a cascade of activations passing through the neural network transfers informations between neurons as they pass through the system before reaching a central node or final node where the information is converted to usable data (tokens, in this instance.)

This is to hopefully allow gradients to develop more quickly as information or values accumulate through each node, enabling backpropagation to have a more pronounced effect on the output of the neural network. Early testing shows a promising effect on loss updates, although more testing and further refinement is likely required.

This is not a perfect recreation of the Mat-mul free architecture as I lack the necessary hardware to fully implement this system, as it requires a large amount of info to be held on hardware which while easy for inference, creates limitations during training. It may be possible to overcome this limitation with smart caching and clearing.. I haven't yet fully completed a model, but based on the loss calculations everything APPEARS to be working correctly if not getting better very fast.

I should note, there are some small bugs due to this being unfinished and numerous revisions during this process. At one point it was abled to use both chunked and unchunked datasets of any type, which I will also upload, but I was not satisfied with the implementation of the architecture.

As it is now, the traning data can be in multiple formats with certain restrictions. .txt files are fine, a parquet or json with text, TEXT, messages will work, as well as one with instruct and output columns which will be concatenated. Or a csv with text columns or instruct/output columns.

To load a dataset, have the files you want to use alone in a folder, and press "select dataset directory" and select the folder. Press "load dataset" to load it. Then, press "select/create tokenized data" and choose no when asked to use existing tokenized data. Enter a new name for a file. Then, you can press "tokenize data" to laod it (after you laod a tokenizer.) If you use a pre-tokenized file you must still press tokenzie daa to load it.

Then you can press start training after adjusting your parameters and loading your model and tokenizer. Note: when pressing stop training, it will have a second dialog box pop up when it actually stops after the current batch is completed so you can save a model if you stop mid training.

You can also create a tokenizer from a vocab.json, I also included a program to convert text files to tokenizers if you would like to try this on your own. I am particularly intrerested in trying to develop a model with a systematically developed tokenizer vocabulary, as most are pretty random and all over the place, but my first attempt to do this was with a character mapped tokenizer and it seemed xtremely limited in what it could do, I haven't tried it with the cascading node neural network yet. 

An inference program is included for testing. Note: if you edit the trainoing or inference program, make sure you are aware the logic of the transformers output a tuple of logits, attention_weights (as it is requried to be able to pass attention between nodes) so make sure that is taken into account if you alter the program

I hope to add on to this and make incremental improvements over time. Also, this is not related to the NodeNet named in literature, so we will stick with Node Neuron Network or NNN, even though Nodenet is a better name.

Citations:

@misc{zhu2024scalablematmulfreelanguagemodeling,
      title={Scalable MatMul-free Language Modeling}, 
      author={Rui-Jie Zhu and Yu Zhang and Ethan Sifferman and Tyler Sheaves and Yiqiao Wang and Dustin Richmond and Peng Zhou and Jason K. Eshraghian},
      year={2024},
      eprint={2406.02528},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.02528}, 
}

@misc{allal2024SmolLM2,
      title={SmolLM2 - with great data, comes great performance}, 
      author={Loubna Ben Allal and Anton Lozhkov and Elie Bakouch and Gabriel Martín Blázquez and Lewis Tunstall and Agustín Piqueres and Andres Marafioti and Cyril Zakka and Leandro von Werra and Thomas Wolf},
      year={2024},
}
