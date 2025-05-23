## AGRec (ACL 2025 Findings)
---
### - Abstract
This paper presents AGRec, adapting LLMs' decoders with graph reasoning for recommendation. 
We augment the decoding logits of LLMs with an auxiliary GNN model to optimize token generation. 
Moreover, we introduce a rankable finite state machine to tackle two challenges: 
(1) adjusting autoregressive generation with discriminative decoders that directly predict user-item similarity, 
and (2) token homogeneity, where LLMs often generate items with similar prefix tokens, narrowing the scope of beam search.

---
### - Instruction

    ### Fine-tune 
    >> bash train.sh

    ### Evaluate
    >> bash test.sh
    >> bash test_without_graph_reasoning.sh

    # Generate logits.pkl for new dataset
    ### 1. Build user and item embeddings (revise main.py and conf files, under ./gnns/)
    >> python preprocessing.py
    >> python main.py

    ### 2. Generate logits via FSMs (under ./)
    >> python LightGCN_embeddings.py
    
---------------------------------------------
    # Visualization
    ### 1. Preparing graph embeddings under ('./data/{dataset}/')
    #@ If you did not prepare them, please download from (https://drive.google.com/drive/folders/1SIQ4ydWIIt-_MZkr_9nJu6Lu2UjpeOir?usp=sharing).
    
    ### 2. Visualizing (under ./visualization)    
    #@ The same color represents users (or items) who have purchased (or been purchased by) the same items (or users).
    #@ It would look better, if you customize colors at Line. 20 of tsne.py and the number of users (or items) in visualizing_user.py (or visualizing_item.py)
    >> python visualizing_item.py
    >> python visualizing_user.py

---
### - Environment
#### (i) Install requirements 
    >> pip install -r requirements.txt
#### (ii) Apply for LLaMA access at [[Hugging Face]](https://huggingface.co/meta-llama/Llama-3.2-1B) 
    >> huggingface-cli login --token [YOUR TOKEN]


### Note
* The checkpoints of LightGCN embeddings for AGRec were uploaded on [[Google Drive]](https://drive.google.com/drive/folders/1SIQ4ydWIIt-_MZkr_9nJu6Lu2UjpeOir?usp=sharing). If there is any problem, , please feel free to contact me at kaysenn@163.com.
* This work focus on exploiting GNN logits to assist LLMs for recommendation. We found FSMs still cannot fully exploit the potential of GNN logits. The follow-up work will improve this.
* After careful check, on the Yelp dataset, users could repeatedly visit the previously visited POIs. The follow-up work will avoid this dataset.

### Acknowledgement
- Code reference: GNNs from [[SELFRec]](https://github.com/Coder-Yu/SELFRec); [[LETTER]](https://github.com/HonghuiBao2000/LETTER); 
- Data reference:  (Instruments, Yelp) [[LETTER]](https://github.com/HonghuiBao2000/LETTER); (Arts, Games) [[LC-Rec]](https://github.com/zhengbw0324/LC-Rec);
