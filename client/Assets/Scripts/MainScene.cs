using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;
using LitAR;
using UITools;

public class MainScene : MonoBehaviour
{
    public GameObject placementIndicator;
    public ARRaycastManager raycastManager;
    public ARLightingReconstructionManager lightingReconManager;

    public GameObject spherePrefab;
    public ReflectionProbe probes;

    // Placement vars
    private Texture2D _managedEnvTex;
    private Pose _placementPose;
    private bool _placementPostIsValid = true;
    private bool _placementIndicatorEnabled = true;
    private GameObject _placedPrefab;
    private GameObject _debugUI;

    private void Awake()
    {
#if UNITY_EDITOR
        Screen.sleepTimeout = SleepTimeout.SystemSetting;
#endif
    }

    // Start is called before the first frame update
    void Start()
    {
        OnScreenConsole.main = new OnScreenConsole("OnScreenConsole");

        GameObject.Find("Init").GetComponent<Button>().onClick.AddListener(OnInitButtonClick);
        GameObject.Find("CaptureNearField").GetComponent<Button>().onClick.AddListener(OnCaptureNearFieldButtonClick);
        GameObject.Find("CaptureFarField").GetComponent<Button>().onClick.AddListener(OnCaptureFarFieldButtonClick);
        GameObject.Find("ResetButton").GetComponent<Button>().onClick.AddListener(OnResetButtonClick);

        GameObject.Find("ToggleUI").GetComponent<Button>().onClick.AddListener(OnToggleUIButtonClick);
        _debugUI = GameObject.Find("DebugUI");
        
        OnScreenConsole.main.Log("System initialized");

        var state = SystemInfo.supportsAccelerometer && SystemInfo.supportsGyroscope;
        OnScreenConsole.main.Log($"Testing acc and gyro {state}");

        lightingReconManager.OnNewEnvironmentMapReceived += OnNewEnvironmentMapReceived;

        // Working with renderings.
        // _managedEnvTex = new Texture2D(512, 256, TextureFormat.RGB24, false);
        // _managedEnvTex = new Texture2D(768, 384, TextureFormat.RGB24, false);
        _managedEnvTex = new Texture2D(1024, 512, TextureFormat.RGB24, false);
        
        var texColors = _managedEnvTex.GetPixels();
        for (var i = 0; i < texColors.Length; i++)
        {
            texColors[i] = Color.red;
        }

        _managedEnvTex.SetPixels(texColors);
        _managedEnvTex.Apply();

        RenderSettings.skybox.mainTexture = _managedEnvTex;
        probes.RenderProbe();
    }

    // Update is called once per frame
    void Update()
    {
        UpdatePlacementPose();
        UpdatePlacementIndicator();
    }

    #region Placement Indicator

    private void UpdatePlacementIndicator()
    {
        if (_placementPostIsValid && _placementIndicatorEnabled)
        {
            placementIndicator.SetActive(true);
            placementIndicator.transform.SetPositionAndRotation(
                _placementPose.position, _placementPose.rotation);
        }
        else
        {
            placementIndicator.SetActive(false);
        }
    }

    private void UpdatePlacementPose()
    {
        Vector3 screenCenter;

        try
        {
            screenCenter = Camera.current.ViewportToScreenPoint(
                new Vector3(0.5f, 0.5f));
        }
        catch
        {
            return;
        }

        var hits = new List<ARRaycastHit>();
        raycastManager.Raycast(screenCenter, hits, TrackableType.Planes);

        _placementPostIsValid = hits.Count > 0;

        if (!_placementPostIsValid) return;

        _placementPose = hits[0].pose;

        var cameraForward = Camera.current.transform.forward;
        var cameraBearing = new Vector3(cameraForward.x, 0, cameraForward.z).normalized;
        _placementPose.rotation = Quaternion.LookRotation(cameraBearing);
    }

    #endregion

    void OnToggleUIButtonClick()
    {
        _debugUI.SetActive(!_debugUI.activeSelf);
    }
    
    void OnResetButtonClick()
    {
        Destroy(_placedPrefab);
        lightingReconManager.ResetAllSessions();
        _placementIndicatorEnabled = true;
    }

    private void OnCaptureNearFieldButtonClick()
    {
        OnScreenConsole.main!.Log("Manually capturing a near field....");
        lightingReconManager.ManuallyCaptureNearField();
    }

    private void OnCaptureFarFieldButtonClick()
    {
        OnScreenConsole.main!.Log("Manually capturing a far field....");
        lightingReconManager.ManuallyCaptureFarField();
    }

    private void OnInitButtonClick()
    {
        if (!_placementIndicatorEnabled)
            OnScreenConsole.main!.Log("Placement not available.");

        var p = _placementPose.position;
        probes.transform.position = p;
        p.y += 0.0501f; // Recon center should be the object center

        lightingReconManager.CreateSession(p);
        _placementIndicatorEnabled = false;

        _placedPrefab = Instantiate(spherePrefab, _placementPose.position, _placementPose.rotation);
    }
    
    private void OnNewEnvironmentMapReceived(object sender, byte[] data)
    {
        _managedEnvTex.LoadImage(data);
        // _managedEnvTex.SetPixelData(data, 0);
        _managedEnvTex.Apply();

        probes.RenderProbe();

        OnScreenConsole.main!.Log("New env.map applied...");
    }
}