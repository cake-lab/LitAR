using System;
using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

namespace LitAR.LightingReconstruction
{
    internal interface IXRCpuImageEncodable
    {
        public void Encode(byte[] packageBytes, int dstOffset);
    }

    internal struct XRDepthImage : IXRCpuImageEncodable
    {
        private XRCpuImage _image;

        public XRDepthImage(AROcclusionManager occlusionManager)
        {
            occlusionManager.TryAcquireEnvironmentDepthCpuImage(out _image);
        }

        public void Encode(byte[] packageBytes, int dstOffset)
        {
            var img = _image.GetPlane(0).data.ToArray();
            Buffer.BlockCopy(img, 0, packageBytes, dstOffset, img.Length);
        }

        public void Dispose()
        {
            _image.Dispose();
        }
    }

    internal struct XRConfidenceFilteredDepthImage : IXRCpuImageEncodable
    {
        private XRCpuImage _depthImage;
        private XRCpuImage _confidenceImage;
        private readonly int _minConfidence;

        public Vector2Int Size()
        {
            return _depthImage.dimensions;
        }

        public XRConfidenceFilteredDepthImage(AROcclusionManager occlusionManager, int minConfidence = 1)
        {
            occlusionManager.TryAcquireEnvironmentDepthCpuImage(out _depthImage);
            occlusionManager.TryAcquireEnvironmentDepthConfidenceCpuImage(out _confidenceImage);
            _minConfidence = minConfidence;
        }

        public void Encode(byte[] packageBytes, int dstOffset)
        {
            var depthValues = _depthImage.GetPlane(0).data.ToArray();
            var confidenceValues = _confidenceImage.GetPlane(0).data;

            for (var i = 0; i < confidenceValues.Length; i++)
            {
                // filter low confidence depth
                // convert to 1000, will be occluded by later calculation on edge
                var c = confidenceValues[i];
                if (c >= _minConfidence) continue;

                // Don't panic, these magic numbers represent number 1000 in float32 format
                depthValues[i * 4 + 0] = 0;
                depthValues[i * 4 + 1] = 0;
                depthValues[i * 4 + 2] = 122;
                depthValues[i * 4 + 3] = 68;
            }

            Buffer.BlockCopy(depthValues, 0, packageBytes, dstOffset, depthValues.Length);
        }

        public void Dispose()
        {
            _depthImage.Dispose();
            _confidenceImage.Dispose();
        }
    }

    internal struct XRRawColorImage : IXRCpuImageEncodable
    {
        private XRCpuImage _image;
        private readonly float _scale;

        private readonly Vector2Int _nativeSize;
        public readonly Vector2Int SampleSize;


        public XRRawColorImage(ARCameraManager cameraManager, Vector2Int sampleSize)
        {
            cameraManager.TryAcquireLatestCpuImage(out _image);

            _nativeSize = _image.dimensions;
            this.SampleSize = sampleSize;
            _scale = this.SampleSize.x / (float)_nativeSize.x;
        }

        public void Encode(byte[] packageBytes, int dstOffset)
        {
            // Currently using nearest sampling, consider upgrade 
            // to bi-linear sampling for better anti-aliasing.
            var planeY = _image.GetPlane(0).data.ToArray();
            for (var v = 0; v < SampleSize.y; v++)
            {
                for (var u = 0; u < SampleSize.x; u++)
                {
                    var iv = (int) (v / _scale);
                    var iu = (int) (u / _scale);
                    packageBytes[dstOffset + v * SampleSize.x + u] = planeY[iv * _nativeSize.x + iu];
                }
            }

            var planeUV = _image.GetPlane(1).data;
            var offsetUV = dstOffset + SampleSize.x * SampleSize.y;
            for (var v = 0; v < SampleSize.y / 2; v++)
            {
                for (var u = 0; u < SampleSize.x / 2; u++)
                {
                    var iv = (int) (v / _scale);
                    var iu = (int) (u / _scale);

                    var sampleOffset = offsetUV + v * SampleSize.x + u * 2;
                    var nativeOffset = iv * _nativeSize.x / 2 * 2 + iu * 2;

                    packageBytes[sampleOffset + 0] = planeUV[nativeOffset + 0];
                    packageBytes[sampleOffset + 1] = planeUV[nativeOffset + 1];
                }
            }
        }

        public void Dispose()
        {
            _image.Dispose();
        }
    }
}
