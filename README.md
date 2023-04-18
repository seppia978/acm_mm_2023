# Unlearning4XAI

## Machine Unlearning technique for Explainable Artificial Intelligence

### Scripts:
- unl_shorter_dataset.py [the script for general unlearning]
- shapely_on_alphas.py [creates the parallel categories plots and visualizes activations and filters]

### Folders:
#### custom_archs
- WTFCNN.py [handler for all the alpha-related routines, from the standard-wtf model conversion to the customized forward step]
- wtflayer.py [implements the abstract class and methods for a general layer]
- wtfconv2d.py [handles all the customizations of a conv2d layer]
- wtfhead.py [handles all the customizations of a transformer layer]

