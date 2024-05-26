# BrainMachineInterfaces
A repo to manage my 4th year project work on Brain Machine Interfaces

This project uses a set of cameras to locate markers/rigid bodies/skeletons in a 3D space space. The data is pre-processed by the Optitrack software, then can be accessed by a pc and used for further analysis. Possible avenues this project could take involve applications such as tracking body movements to predict mood and enabling participants to communicate and map their spatial actions to machines.

Week 1 and 2 has focused on looking at possible ideas of where to take this project and learning to capture data for rigid bodies and markers. Week 3 is about learning to stream the data from optitrack for live visualisation and the image below shows how this will be tackled

![image](Images/DisplayingStreamedDataPlan.png)

Week 3 and 4 focuses on streaming and rendeering the data from optitrack. Rigid bodies deemed important to decoding motion are rendered and identified. At the moment, when I perform jumping jacks the delay is roughly 180 degrees out of phase.

Week 5 and 6 is about measuring the delay time of the rendering and striving to reduce it. Also we aim to show visual motion of a cursor on a screen to show the basics of human interaction wth the screen. people will expect the cursor to move at different speeds and so this is one parameter that can be adjusted. We will also experiment controlling with different rigid bodies. We will also measure the reaction times. look up standard reach task as it is comparable with literature.

Week 7 and 8 will then be about creating a virtual pendulum and creating a feedback loop allowing the user to perform certain actions to balance it. Then the aim will be on creating a decoder to better control the pendulum. Extensive data will need to be collected for this.

In the holidays a further experiment can be designed that will interrogate the user's control strategies in an improved manner and will also offer big data analysis methods to dynamically alter the control.