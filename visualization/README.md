## AGRec (ACL 2025 Findings)

---
### - Instruction

    # Visualization
    ### 1. Preparing graph embeddings under ('./data/{dataset}/')
    #@ If you did not prepare them, please download from (https://drive.google.com/drive/folders/1SIQ4ydWIIt-_MZkr_9nJu6Lu2UjpeOir?usp=sharing).
    
    ### 2. Visualizing (under ./visualization)    
    #@ The same color represents users (or items) who have purchased (or been purchased by) the same items (or users).
    #@ It would look better, if you customize colors at Line. 20 of tsne.py and the number of users (or items) in visualizing_user.py (or visualizing_item.py)
    >> python visualizing_item.py
    >> python visualizing_user.py
