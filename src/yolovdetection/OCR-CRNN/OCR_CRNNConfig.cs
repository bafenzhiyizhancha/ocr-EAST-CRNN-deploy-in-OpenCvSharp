

namespace OCR_CRNN
{
    /// <summary>
    /// 参数
    /// </summary>
    public class OCR_CRNNConfig
    {
        /// <summary>
        /// 文本检测模型文件路径
        /// </summary>
        public string DetectorModelPath { set; get; }


        /// <summary>
        /// 文本识别模型文件路径
        /// </summary>
        public string RecognitionModelPath { set; get; }

        /// <summary>
        /// 设置分类
        /// </summary>
        public string Alphabet { set; get; }

        /// <summary>
        /// 置信度阈值
        /// </summary>
        public float ConfThreshold { set; get; }

        /// <summary>
        /// nms 阈值
        /// </summary>
        public float NmsThreshold { set; get; }

        /// <summary>
        /// 模型要求的的图片大小
        /// </summary>
        public int ImgWidth { set; get; }

        /// <summary>
        ///模型要求的的图片大小
        /// </summary>
        public int ImgHight { set; get; }

        /// <summary>
        /// 是否显示图像
        /// </summary>
        public bool IsDraw { set; get; }

        public OCR_CRNNConfig()
        {
            IsDraw = true;
        }
    }
}
