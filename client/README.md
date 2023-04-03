# LitAR Reference Application

This repo is a reference implementation of an AR app that is built with LitAR.
Note that we include the LitAR APIs via the following files:
- `./Assets/Scripts/LitARR`: core framework components

Below we first describe the basic steps to run the reference AR application on a Unity-compatible computer, followed by instructions to run on compatible iOS devices.

## Basic Setup

Before setting up the reference application, please make sure you have setup the [LitAR server](../server/) with a publicly-accessible network address.
Then, update the `Endpoint` constant (in `Assets/Scripts/LitAR/Network/ARLightingReconstructionManager.cs`) to point to the server address. Below is an example when we set the `Endpoint` to point to our lab's server.

```csharp
private const string Endpoint = "http://cake-graphics.dyn.wpi.edu:8550";
```

To setup the LitAR reference application, please first download the Unity3D and load `client` directory as a project.
Once the project finishes loading, use the Play button on the center top of Unity editor to start the AR application.

After the application starts, please follow the Unity editor's console log information to identify the appropriate Unity's application persistent data path.

## iOS Deployment

To deploy the application to iOS device, first change the project build settings to `iOS` to generate an iOS project.
Then, compile and deploy the project to an iOS device.
