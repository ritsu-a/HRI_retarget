import pickle 

with open("/home/pengyang/codebase/H1_RL/data/motion/galbot/SG/058_clip_semantic_results.pickle", "rb") as file:
    data = pickle.load(file)

data["angles"] = data["angles"][120:]

with open("/home/pengyang/codebase/H1_RL/data/motion/galbot/SG/058_clip_semantic_results_crop.pickle", "wb") as file:
    pickle.dump(data, file)
