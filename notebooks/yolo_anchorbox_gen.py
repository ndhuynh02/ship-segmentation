import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import json
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from src.utils.kmeans import kmeans
from src.data.airbus.components.airbus import AirbusDataset


def main():
    data = AirbusDataset(undersample=-1, subset=0, bbox_format="midpoint")
    boxes = []
    result = {}

    print("Collecting Data")
    for i in tqdm(range(len(data))):
        _, _, box, _ = data[i]

        boxes += box[:, -2:].tolist()

    boxes = np.array(boxes)

    clusterList = [3, 4, 5, 6, 7, 8, 9]
    mean_distances = []

    print("Estimating anchor boxes")
    for cluster_k in tqdm(clusterList):
        anchors, distances = kmeans(boxes, cluster_k)
        indxs = np.argmax(distances, axis=1)
        filtered_distances = []
        for i, distance in enumerate(distances):
            filtered_distances.append(distance[indxs[i]].item())
        mean_distances.append(np.mean(filtered_distances))
        result[cluster_k] = anchors.tolist()
        
    plt.plot(clusterList, mean_distances)
    plt.scatter(clusterList, mean_distances)
    plt.title("Mean IoU Score")
    plt.xlabel("Number of clusters")
    plt.ylabel("IoU score")
    plt.savefig("anchor iou graph.jpg")

    with open("anchor boxes.json", "w") as outfile: 
        json.dump(result, outfile, indent=4)


if __name__ == "__main__":
    main()