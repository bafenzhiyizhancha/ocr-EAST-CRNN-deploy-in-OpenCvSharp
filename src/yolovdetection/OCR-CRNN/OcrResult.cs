
using OpenCvSharp;

namespace OCR_CRNN
{
    /// <summary>
    /// 结果存储格式
    /// </summary>
    public class OcrResult
    {
        /// <summary>
        /// 字符区域的位置
        /// </summary>
        public Point2f[] Vertices { set; get; }

        /// <summary>
        /// 识别出的字符
        /// </summary>
        public string Test { set; get; }

        internal static OcrResult Add(Point2f[] points,string text)
        {
            return new OcrResult()
            {
                Vertices = points,
                Test = text
            };
        }
    }
}
