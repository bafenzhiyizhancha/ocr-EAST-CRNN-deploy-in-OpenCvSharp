
using OpenCvSharp;
using OpenCvSharp.Dnn;

namespace OCR_CRNN
{
    /// <summary>
    /// 模型持久化变量
    /// 对象：在检测中涉及到需要较大内存的变量
    /// 目的：将这些变量的从检测流程中提取出，移动到初始化中，
    /// 将整个流程分为初始化与检测两个部分
    /// 一次初始化，多次使用，加速检测部分.
    /// </summary>
    public class ModelKeepValue
    {
        /// <summary>
        /// 定位模型
        /// </summary>
        public Net DetectoterNet { set; get; }  

        /// <summary>
        /// 检测头对应的输出层名字
        /// </summary>
        public string[] OutputBlobNames { set; get; }

        /// <summary>
        /// 模型检测头
        /// </summary>
        public Mat[] OutputBlobs { set; get; }

        /// <summary>
        /// 识别模型
        /// </summary>
        public Net RecognitionNet { set; get; }   
    }
}
