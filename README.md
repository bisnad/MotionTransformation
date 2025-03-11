# AI-Toolbox - Motion Transformation

The "Motion Transformation" category of the AI-Toolbox contains a collection of python-based generative machine learning models that can be trained on motion capture data. These models operate as Autoencoders that take as input as short motion excerpt, encode these excerpts into a compressed representation in latent space, and decode the compressed representation back into a short motion excerpt.  At the moment, the motion transformation tools implement two types of Autoencoders: Adversarial Autoencoders and Variational Autoencoders. Adversarial Autoencoders provide better reconstruction accuracy than Variational Autoencoders, while Variational Autoencoders exhibit fewer artifacts when directly manipulating encodings. Both versions of the motion transformation tools provide the possibility to transform motions by manipulating their compressed representations. By varying the amount by which the compressed representations deviate from an  original encoded motion, the level of similarity and dissimilarity between original motion and the decoded motion can be controlled.  

The following tools are available:

- [aae-rnn](aae-rnn)

  A Python-based tool for training a motion-transformation model on motion capture data. The model is based on an Adversarial Autoencoder. 

- [aae-rnn_interactive](aae-rnn_interactive)

  A Python-based tool that employ a previously trained motion transformation model to generate synthetic motions in real-time. Here, the short motion excerpts that are encoded and decoded stem from a motion capture recording. 

- [aae-rnn_interactive_pos](aae-rnn_interactive_pos)

  A Python-based tool that employs a previously trained motion transformation model to generate synthetic motions in real-time. Contrary to the other tools, this model works with skeleton joint positions that have been obtained by using one of the 2D or 3D pose estimation tools provided by the AI-Toolbox (see MotionAnalysis/PoseEstimation). 


- [vae-rnn](vae-rnn)

  A Python-based tool for training a motion-transformation model on motion capture data. The model is based on a Variational Autoencoder. 

- [vae-rnn_interactive](vae-rnn_interactive)

  A Python-based tool that employ a previously trained motion transformation model to generate synthetic motions in real-time. Here, the short motion excerpts that are encoded and decoded stem from a motion capture recording. 

- [vae-rnn_interactive_live_mocap](vae-rnn_interactive_live_mocap)

  A Python-based tool that employs a previously trained motion transformation model to generate synthetic motions in real-time. Here, the short motion excerpt that are encoded and decoded  is live captured


- [vae-rnn_interactive_pos](vae-rnn_interactive_pos)

  A Python-based tool that employs a previously trained motion transformation model to generate synthetic motions in real-time. Contrary to the other tools, this model works with skeleton joint positions that have been obtained by using one of the 2D or 3D pose estimation tools provided by the AI-Toolbox (see MotionAnalysis/PoseEstimation). 
