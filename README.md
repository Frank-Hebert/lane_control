# lane_control

Add this package like you would with any other packages. In your dt-exercises folder, go to your packages location (.../exercises_ws/src/) and git clone this github.

To activate the code on your robot, go to an exercises folder of your choice and run these commands :

```console
dts exercises build
dts exercises test --duckiebot_name ![ROBOT_NAME] --local --pull
```

Then, open your browser and go to your local host adresse (http://localhost:8087/). From this NoVNC environement, launch the joystick controller and press "A" to activate the following procedure.


**Expected behavior :**

If your duckiebot detect a circles grid pattern (rear bumper of a duckiebot), it will go toward it and will park itself behind the other duckiebot.
