using System;
using UnityEngine;
using UnityEngine.XR.ARSubsystems;

namespace LitAR.LightingReconstruction
{
    internal struct SessionInitPackage
    {
        private const byte PackageIdentifier = 0b0000_0000;

        public int numOfViews;
        public int expTimeWindow; // in milliseconds
        public int nearFieldSize; // in millimeters

        public float AmbientAvgColorBrightness;
        public float AmbientAvgColorTemperature;

        public XRCameraIntrinsics k;
        public Vector2Int depthNativeSize;
        public Vector2Int colorNativeSize;
        public Vector2Int colorDenseSamplingSize;
        public Vector2Int colorSparseSamplingSize;

        public byte[] EncodeToBytes()
        {
            const int headerLength = 1;
            const int configLength = 3 * sizeof(int);
            const int ambientInfoLength = 2 * sizeof(float);
            const int matrixKLength = 4 * sizeof(float);
            const int imgSizesLength = 6 * sizeof(int);

            var pkgBytes = new byte[headerLength
                                    + configLength
                                    + ambientInfoLength
                                    + matrixKLength
                                    + imgSizesLength];

            pkgBytes[0] = PackageIdentifier;
            var offset = headerLength;


            Buffer.BlockCopy(new[]
            {
                numOfViews,
                expTimeWindow,
                nearFieldSize
            }, 0, pkgBytes, offset, configLength);
            offset += configLength;


            Buffer.BlockCopy(new[]
            {
                AmbientAvgColorTemperature,
                AmbientAvgColorBrightness
            }, 0, pkgBytes, offset, ambientInfoLength);
            offset += ambientInfoLength;


            var sampleToNativeRatio = colorDenseSamplingSize.x / (float) colorNativeSize.x;
            Buffer.BlockCopy(new[]
            {
                k.focalLength.x * sampleToNativeRatio,
                k.focalLength.y * sampleToNativeRatio,
                k.principalPoint.x * sampleToNativeRatio,
                k.principalPoint.y * sampleToNativeRatio
            }, 0, pkgBytes, offset, matrixKLength);
            offset += matrixKLength;


            Buffer.BlockCopy(new[]
            {
                depthNativeSize.x,
                depthNativeSize.y,
                colorDenseSamplingSize.x,
                colorDenseSamplingSize.y,
                colorSparseSamplingSize.x,
                colorSparseSamplingSize.y
            }, 0, pkgBytes, offset, imgSizesLength);

            return pkgBytes;
        }
    }

    internal struct NearFieldKeyFramePackage
    {
        private const byte PackageIdentifier = 0b0001_0000;

        public byte ViewIndex;
        public Matrix4x4 MatrixTRS;

        public XRRawColorImage ColorImage;
        public XRConfidenceFilteredDepthImage DepthImage;

        public byte[] EncodeToBytes(Vector3 reconPosition)
        {
            const int headerLength = 1;
            const int viewIndexLength = 1 * sizeof(int);
            const int matrixTRSLength = 3 * 4 * sizeof(float);

            var cSize = ColorImage.SampleSize;
            var colorBodyLength = (int) (cSize.x * cSize.y * 1.5);
            var dSize = DepthImage.Size();
            var depthBodyLength = dSize.x * dSize.y * sizeof(float);

            // Start encoding data into one byte buffer
            var offset = 0;
            var pkgBytes = new byte[headerLength
                                    + viewIndexLength
                                    + matrixTRSLength
                                    + colorBodyLength
                                    + depthBodyLength];

            // Fill header bytes into the package
            pkgBytes[0] = PackageIdentifier;
            pkgBytes[1] = ViewIndex;
            offset = headerLength + viewIndexLength;

            // Fill TRS matrix
            var m = MatrixTRS;
            var trsMatrix = new[] // Only 3x4 matrix is needed.
            {
                m.m00, m.m01, m.m02, m.m03 - reconPosition.x,
                m.m10, m.m11, m.m12, m.m13 - reconPosition.y,
                m.m20, m.m21, m.m22, m.m23 - reconPosition.z
            };
            Buffer.BlockCopy(trsMatrix, 0, pkgBytes, offset, matrixTRSLength);
            offset += matrixTRSLength;


            // Fill color image bytes
            ColorImage.Encode(pkgBytes, offset);
            ColorImage.Dispose();
            offset += colorBodyLength;


            // Fill depth image
            DepthImage.Encode(pkgBytes, offset);
            DepthImage.Dispose();


            return pkgBytes;
        }
    }

    internal struct FarFieldKeyFramePackage
    {
        private const byte PackageIdentifier = 0b0001_0001;
        
        public Matrix4x4 MatrixTRS;
        public XRRawColorImage ColorImage;
        
        public byte[] EncodeToBytes()
        {
            const int headerLength = 1;
            const int matrixTRSLength = 3 * 3 * sizeof(float);

            var cSize = ColorImage.SampleSize;
            var colorBodyLength = (int) (cSize.x * cSize.y * 1.5);

            // Start encoding data into one byte buffer
            var offset = 0;
            var pkgBytes = new byte[headerLength
                                    + matrixTRSLength
                                    + colorBodyLength];

            // Fill header bytes into the package
            pkgBytes[0] = PackageIdentifier;
            offset = headerLength;

            // Fill TRS matrix
            var m = MatrixTRS;
            var trsMatrix = new[] // Only 3x4 matrix is needed.
            {
                m.m00, m.m01, m.m02,
                m.m10, m.m11, m.m12,
                m.m20, m.m21, m.m22
            };
            Buffer.BlockCopy(trsMatrix, 0, pkgBytes, offset, matrixTRSLength);
            offset += matrixTRSLength;


            // Fill color image bytes
            ColorImage.Encode(pkgBytes, offset);
            ColorImage.Dispose();

            return pkgBytes;
        }
    }
}