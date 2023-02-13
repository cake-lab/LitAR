using JetBrains.Annotations;
using UnityEngine;
using UnityEngine.UI;

namespace UITools
{
    public class OnScreenConsole
    {
        [CanBeNull] public static OnScreenConsole main;
        
        private Text txtObj;
    
        public OnScreenConsole(string gameObjName)
        {
            txtObj = GameObject.Find(gameObjName).GetComponent<Text>();
        }

        public void Log(string message)
        {
            txtObj.text += $"{message}\n";
        }
    }
}