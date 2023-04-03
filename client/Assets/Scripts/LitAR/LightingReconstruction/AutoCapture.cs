using System;
using System.Collections.Generic;
using System.Linq;
using UITools;
using UnityEngine;
using UnityEngine.XR.ARFoundation;

namespace LitAR.LightingReconstruction
{
    class TimedViewPose
    {
        public DateTime Time;
        public Pose Pose;
    }

    // Multi-sensor driven auto capturing
    internal class AutoCaptureController
    {
        public bool Enabled = false;
        public Vector3 ReconPosition;
        public ARLightingReconstructionManager ReconManager;

        private DateTime _timer;
        private readonly SessionInitPackage _config;

        private byte _iView = 0;
        private float _currentColorTemp = 0;
        private readonly Vector3[] _vvForward;
        private readonly Vector3[] _vvPosition;

        private readonly Queue<Pose> _devicePoseQueue;
        private readonly List<TimedViewPose> _viewPoses;

        public AutoCaptureController(SessionInitPackage config)
        {
            _config = config;
            _timer = new DateTime();

            _vvForward = new Vector3[_config.numOfViews];
            _vvPosition = new Vector3[_config.numOfViews];

            _devicePoseQueue = new Queue<Pose>(5);
            _viewPoses = new List<TimedViewPose>();
        }

        public byte[] EvaluateFrame(ARCameraFrameEventArgs frameData)
        {
            if (!Enabled) return null;

            if (!TestDeviceStable()) return null;

            if (!TestTimer()) return null;

            return TestReconPosVisible()
                ? new NearFieldKeyFramePackage
                {
                    ViewIndex = (byte) GetCurrentViewIndex(),
                    MatrixTRS = Camera.main!.transform.localToWorldMatrix,
                    ColorImage = new XRRawColorImage(ReconManager.cameraManager, _config.colorDenseSamplingSize),
                    DepthImage = new XRConfidenceFilteredDepthImage(ReconManager.occlusionManager, 2)
                }.EncodeToBytes(ReconPosition)
                : new FarFieldKeyFramePackage
                {
                    MatrixTRS = Camera.main!.transform.localToWorldMatrix,
                    ColorImage = new XRRawColorImage(ReconManager.cameraManager, _config.colorSparseSamplingSize)
                }.EncodeToBytes();
        }

        public byte[] CaptureNearField()
        {
            OnScreenConsole.main!.Log("Capturing a near field...");

            return new NearFieldKeyFramePackage
            {
                ViewIndex = (byte) (_iView++ % _config.numOfViews),
                MatrixTRS = Camera.main!.transform.localToWorldMatrix,
                ColorImage = new XRRawColorImage(ReconManager.cameraManager, _config.colorDenseSamplingSize),
                DepthImage = new XRConfidenceFilteredDepthImage(ReconManager.occlusionManager, 2)
            }.EncodeToBytes(ReconPosition);
        }

        public byte[] CaptureFarField()
        {
            OnScreenConsole.main!.Log("Capturing a far field...");

            return new FarFieldKeyFramePackage
            {
                MatrixTRS = Camera.main!.transform.localToWorldMatrix,
                ColorImage = new XRRawColorImage(ReconManager.cameraManager, _config.colorSparseSamplingSize)
            }.EncodeToBytes();
        }

        private int GetCurrentViewIndex()
        {
            // return (_iView++ % _config.numOfViews);
            
            var t = Camera.main!.transform;
            var p = new Pose(t.position, t.rotation);

            // If a historical related pose were found
            for (var i = 0; i < _viewPoses.Count; i++)
            {
                var vp = _viewPoses[i];

                var c1 = (p.position - vp.Pose.position).magnitude < 0.1;
                var c2 = Quaternion.Angle(p.rotation, vp.Pose.rotation) < Math.PI / 10;

                if (!c1 || !c2) continue;

                _viewPoses[i].Pose = p;
                _viewPoses[i].Time = DateTime.Now;

                return i;
            }

            // If there are still empty spaces
            if (_viewPoses.Count < _config.numOfViews)
            {
                _viewPoses.Add(new TimedViewPose
                {
                    Time = DateTime.Now,
                    Pose = p
                });

                return _viewPoses.Count - 1;
            }

            // Replace the oldest one
            var s = _viewPoses
                .Select((v, i) => new KeyValuePair<int, TimedViewPose>(i, v))
                .OrderBy(v => v.Value.Time)
                .ToList();
            var idx = s[0].Key;
            _viewPoses[idx].Pose = p;
            _viewPoses[idx].Time = DateTime.Now;

            return idx;
        }

        private bool TestDeviceStable()
        {
            var t = Camera.main!.transform;
            var p = new Pose(t.position, t.rotation);

            var res = true;

            foreach (var q in _devicePoseQueue)
            {
                res = (q.position - p.position).magnitude < 0.05
                      && Quaternion.Angle(q.rotation, p.rotation) < Math.PI / 36;
                if (!res) break;
            }

            _devicePoseQueue.Enqueue(p);

            if (_devicePoseQueue.Count > 5) _devicePoseQueue.Dequeue();

            return res;
        }

        private bool TestTimer()
        {
            var now = DateTime.Now;

            var delta = (now - _timer).TotalMilliseconds;
            var res = delta > _config.expTimeWindow;

            if (res) _timer = now;

            return res;
        }

        private bool TestAmbientLightSensor(ARLightEstimationData data)
        {
            if (!data.averageColorTemperature.HasValue) return false;
            var t = data.averageColorTemperature.Value;
            var res = t - _currentColorTemp > 1000;

            if (res) _currentColorTemp = t;

            return res;
        }

        private bool TestMotionSensor(bool reconPosInView)
        {
            var ct = Camera.main!.transform;

            // Delta rotation > 30 degrees
            var cr = Vector3.Dot(ct.forward, _vvForward[_iView]) < Math.PI / 6;

            if (!reconPosInView) return cr;

            // Delta position > 20 cm
            var cp = (_vvPosition[_iView] - ct.position).sqrMagnitude > 0.2;

            if (!(cr || cp)) return false;

            _iView = (byte) ((_iView + 1) % _config.numOfViews);
            _vvForward[_iView] = ct.forward;
            _vvPosition[_iView] = ct.position;
            return true;
        }

        private bool TestReconPosVisible()
        {
            var uv = Camera.main!.WorldToViewportPoint(ReconPosition);

            return uv.z > 0 && uv.x is < 1 and > 0 && uv.y is < 1 and > 0;
        }
    }
}