using System;
using System.Collections.Generic;
using System.Diagnostics;
using LitAR.LightingReconstruction;
using NativeWebSocket;
using UITools;
using UnityEngine;
using UnityEngine.XR.ARFoundation;
using Debug = UnityEngine.Debug;

namespace LitAR
{
    public class ARLightingReconstructionManager : MonoBehaviour
    {
        public ARCameraManager cameraManager;
        public AROcclusionManager occlusionManager;
        public string serverUrl;
        public float colorToDepthSamplingRatio = 1.0f;

        private WebSocket _wsClient;
        private ARLightEstimationData _ambientInfo;
        private AutoCaptureController _tmpController;
        private readonly Dictionary<string, AutoCaptureController> _captureControllers = new();

        private Stopwatch _e2eTimer;

        private const bool EnableAutoCapture = false;

        public event EventHandler<byte[]> OnNewEnvironmentMapReceived;

        private async void Start()
        {
            _wsClient = new WebSocket(serverUrl);
            _wsClient.OnMessage += WebSocketClientOnOnMessage;
            _wsClient.OnOpen += () => { Debug.Log("Connection open!"); };
            _wsClient.OnError += (e) => { Debug.Log("Error! " + e); };
            _wsClient.OnClose += (e) =>
            {
                OnScreenConsole.main!.Log("Connection closed...");
            };

            cameraManager.frameReceived += CameraManagerOnFrameReceived;
            
            // waiting for messages, this line will block execution
            await _wsClient.Connect();
        }

        private void Update()
        {
#if !UNITY_WEBGL || UNITY_EDITOR
            _wsClient.DispatchMessageQueue();
#endif
        }

        public async void CreateSession(Vector3 reconPosition)
        {
            if (_captureControllers.Count > 0)
            {
                OnScreenConsole.main!.Log("Adding more than 1 controllers, not supported yet.");
                return;
            }

            // Get runtime info
            cameraManager.TryAcquireLatestCpuImage(out var colorImage);
            occlusionManager.TryAcquireEnvironmentDepthCpuImage(out var depthImage);
            cameraManager.TryGetIntrinsics(out var k);

            // Create session initialization data
            
            // High quality setting
            var initData = new SessionInitPackage
            {
                numOfViews = 5,
                expTimeWindow = 300,
                nearFieldSize = 2 * 100,
                AmbientAvgColorBrightness = _ambientInfo.averageBrightness!.Value,
                AmbientAvgColorTemperature = _ambientInfo.averageColorTemperature!.Value,
                k = k,
                depthNativeSize = depthImage.dimensions,
                colorNativeSize = colorImage.dimensions,
                colorDenseSamplingSize = new Vector2Int(
                    (int) (depthImage.dimensions.x * 4),
                    (int) (depthImage.dimensions.y * 4)),
                colorSparseSamplingSize = new Vector2Int(32, 24)
            };
            
            // Medium
            // var initData = new SessionInitPackage
            // {
            //     numOfViews = 4,
            //     expTimeWindow = 300,
            //     nearFieldSize = 2 * 100,
            //     AmbientAvgColorBrightness = _ambientInfo.averageBrightness!.Value,
            //     AmbientAvgColorTemperature = _ambientInfo.averageColorTemperature!.Value,
            //     k = k,
            //     depthNativeSize = depthImage.dimensions,
            //     colorNativeSize = colorImage.dimensions,
            //     colorDenseSamplingSize = new Vector2Int(
            //         (int) (depthImage.dimensions.x * 2),
            //         (int) (depthImage.dimensions.y * 2)),
            //     colorSparseSamplingSize = new Vector2Int(32, 24)
            // };
            
            // Low quality setting
            // var initData = new SessionInitPackage
            // {
            //     numOfViews = 3,
            //     expTimeWindow = 300,
            //     nearFieldSize = 2 * 100,
            //     AmbientAvgColorBrightness = _ambientInfo.averageBrightness!.Value,
            //     AmbientAvgColorTemperature = _ambientInfo.averageColorTemperature!.Value,
            //     k = k,
            //     depthNativeSize = depthImage.dimensions,
            //     colorNativeSize = colorImage.dimensions,
            //     colorDenseSamplingSize = new Vector2Int(
            //         (int) (depthImage.dimensions.x * 1),
            //         (int) (depthImage.dimensions.y * 1)),
            //     colorSparseSamplingSize = new Vector2Int(32, 24)
            // };

            _tmpController = new AutoCaptureController(initData)
            {
                Enabled = false,
                ReconManager = this,
                ReconPosition = reconPosition
            };

            // Send bytes to edge
            await _wsClient.Send(initData.EncodeToBytes());

            OnScreenConsole.main?.Log($"Service Address: {serverUrl}");
            OnScreenConsole.main?.Log("FusionAR Initialized...");
        }

        public void DestroySession(string sessionID)
        {
            _captureControllers.Remove(sessionID);
        }

        public void ResetAllSessions()
        {
            _captureControllers.Clear();
            _wsClient.Connect();
        }

        public async void ManuallyCaptureNearField()
        {
            _e2eTimer = Stopwatch.StartNew();
            
            foreach (var controllers in _captureControllers.Values)
            {
                await _wsClient.Send(controllers.CaptureNearField());
            }
        }

        public async void ManuallyCaptureFarField()
        {
            _e2eTimer = Stopwatch.StartNew();

            foreach (var controllers in _captureControllers.Values)
            {
                await _wsClient.Send(controllers.CaptureFarField());
            }
            
        }

        private void CameraManagerOnFrameReceived(ARCameraFrameEventArgs obj)
        {
            // Check for needed vars
            var c = obj.lightEstimation.averageColorTemperature.HasValue
                    && obj.lightEstimation.averageBrightness.HasValue;
            if (!c) return;


            _ambientInfo = obj.lightEstimation;


            // Evaluate each auto capture controller
            foreach (var controller in _captureControllers.Values)
            {
                var keyframeData = controller.EvaluateFrame(obj);
                if (keyframeData == null) continue;

                controller.Enabled = false;
                OnScreenConsole.main!.Log("Sending a keyframe...");
                _wsClient.Send(keyframeData);
            }
        }

        private void WebSocketClientOnOnMessage(byte[] data)
        {
            var header = data[0];
            var dataBody = new Span<byte>(data, 1, data.Length - 1);

            switch (header)
            {
                case 0b_0000_0001:
                    OnSessionInitializationFinished(dataBody);
                    break;
                case 0b_0001_0000:
                    OnNewEnvironmentMapReceived?.Invoke(this, dataBody.ToArray());

                    OnScreenConsole.main!.Log($"Runtime e2e: {_e2eTimer.ElapsedMilliseconds}ms");

                    // TODO: enable only the current session controller
                    foreach (var controller in _captureControllers.Values)
                    {
                        controller.Enabled = EnableAutoCapture;
                    }
                    break;
            }
        }

        private void OnSessionInitializationFinished(Span<byte> data)
        {
            var s = System.Text.Encoding.Default.GetString(data);
            OnScreenConsole.main!.Log($"Session init finished, s_id {s}");

            _captureControllers.Add(s, _tmpController);
            _tmpController.Enabled = EnableAutoCapture;
        }
        
        private void OnDisable()
        {
            cameraManager.frameReceived -= CameraManagerOnFrameReceived;
        }

        private async void OnApplicationQuit()
        {
            await _wsClient.Close();
        }
    }
}