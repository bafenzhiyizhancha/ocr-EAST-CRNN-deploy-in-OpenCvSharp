
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace OCR_CRNN
{
    /// <summary>
    /// 文本检测用的是EAST，文本识别用的是CRNN
    /// </summary>
    public class OCR_CRNNDetect
    {
        private readonly OCR_CRNNConfig _config;    //模型检测参数
        private readonly ModelKeepValue _model;     //模型持久化变量

        /// <summary>
        /// 初始化
        /// </summary>
        /// <param name="dectecteoPath"> 检测模型的路径</param>
        /// <param name="recognitionPath">识别模型的路径</param>
        /// <param name="labelsFile">标签路径</param>
        /// <param name="imgWidth">模型要求的的图片大小</param>
        /// <param name="imgHigh">模型要求的的图片大小</param>
        /// <param name="threshold">置信度阈值</param>
        /// <param name="nms">nms 阈值</param>
        public OCR_CRNNDetect(string dectecteoPath, string recognitionPath, string labelsFile,
                       int imgWidth = 320, int imgHigh = 320, float threshold = 0.5f, float nms = 0.4f)
        {
            _config = new OCR_CRNNConfig
            {
                DetectorModelPath = dectecteoPath,
                RecognitionModelPath = recognitionPath,
                Alphabet = ReadLabels(labelsFile),
                ImgWidth = imgWidth,
                ImgHight = imgHigh,
                ConfThreshold = threshold,
                NmsThreshold = nms
            };
            _model = new ModelKeepValue();
            Init();
        }


        #region 总流程
        /// <summary>
        /// 检测
        /// </summary>
        /// <param name="imgpath">图像路径</param>
        /// <returns></returns>
        public OcrResult[] Detect(string imgpath)
        {
            Mat img = Cv2.ImRead(imgpath);
            return Process(img);
        }

        /// <summary>
        /// 检测
        /// </summary>
        /// <param name="img"></param>
        /// <returns></returns>
        public OcrResult[] Detect(System.Drawing.Bitmap img)
        {
            Mat mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(img);
            return Process(mat);
        }

        /// <summary>
        /// 初始化
        /// </summary>
        /// <returns></returns>
        private void Init()
        {
            InitDetectorModel();  
            InitRecognitionModel();
        }
        #endregion

        #region 定位模型
        /// <summary>
        ///  文本检测模型图片预处理
        /// </summary>
        /// <param Mat img</param>
        /// <returns></returns>
        private Mat DetectoterPretreatment(Mat img)
        {
            double scale = 1.0;                      
            Size size = new Size(_config.ImgWidth, _config.ImgHight);

            //数据居中：mean常规操作
            Scalar scalar = new Scalar(123.68, 116.78, 103.94);

            //网络结果是使用同大小一图像进行训练的所以需要将图像转为对应的大小
            Mat blob = CvDnn.BlobFromImage(img, scale, size, scalar, true, false);

            return blob;
        }

        /// <summary>
        /// 初始化文本检测打开pd模型
        /// </summary>
        private Net InitializeDetectoterModel(string pathModel)
        {
            if (!File.Exists(pathModel))
                throw new FileNotFoundException("The file model has not found", pathModel);

            Net net = CvDnn.ReadNet(pathModel);
            if (net == null || net.Empty())
                throw new Exception("The model has not yet initialized or is empty.");

            //读入模型和设置
            net.SetPreferableBackend(Backend.OPENCV);     // 3:DNN_BACKEND_OPENCV 
            net.SetPreferableTarget(Target.CPU);          //dnn target cpu

            return net;
        }

        /// <summary>
        /// 检测的后处理
        /// </summary>
        /// <param name="image"></param>
        /// <param name="results">1、字符位置；2、置信度分数</param>
        /// <returns></returns>
        private OcrResult[] Postprocess(Mat image, Mat[] results)
        {
            var ocrResult = new List<OcrResult>();         //存储检测结果

            Mat scores = results[0];                      //置信度得分
            Mat geometry = results[1];                    //字符位置

            Decode(scores, geometry, out var boxes, out var confidences);

            // Apply non-maximum suppression procedure.
            CvDnn.NMSBoxes(boxes, confidences, _config.ConfThreshold, _config.NmsThreshold, out int[] indices);

            Point2f ratio = new Point2f((float)image.Cols /_config.ImgWidth,(float)image.Rows / _config.ImgHight);
            foreach (var i in indices)
            {
                var box = boxes[i];
                Point2f[] vertices = box.Points();

                //由于该网络检查包含角度，所以输出的区域可能不是水平的矩形
                //恢复原图大小
                for (int j = 0; j < 4; ++j)
                {
                    vertices[j].X *= ratio.X;
                    vertices[j].Y *= ratio.Y;
                }

                //获取字符切片图像
                GetCropImage(image, vertices, out Mat crop);
                string text = RecognitionProcess(crop);

                //存储检测结果
                ocrResult.Add(OcrResult.Add(vertices, text));

                if (true == _config.IsDraw)
                {
                    Draw(image,vertices,text);
                }
               
            }

            return ocrResult.ToArray();        // 返回检测结果
        }

        /// <summary>
        /// 解码
        /// 该模型需要进行解码操作：属于ocr模型中编码-解码网络结构 
        /// </summary>
        /// <param name="scores">置信度分数</param>
        /// <param name="geometry">字符位置</param>
        /// <param name="boxes">输出 最终位置框</param>
        /// <param name="confidences">输出：最终得分</param>
        private void Decode(Mat scores, Mat geometry, out IList<RotatedRect> boxes, out IList<float> confidences)
        {
            confidences = new List<float>();                       //可能对象的置信度集合
            boxes       = new List<RotatedRect>();                 //可能对象的方框集合

            if ((scores == null || scores.Dims != 4 || scores.Size(0) != 1 || scores.Size(1) != 1) ||
                (geometry == null || geometry.Dims != 4 || geometry.Size(0) != 1 || geometry.Size(1) != 5) ||
                (scores.Size(2) != geometry.Size(2) || scores.Size(3) != geometry.Size(3)))
            {
                return;
            }

            var height = scores.Size(2);                            //图像的宽高
            var width = scores.Size(3);

            for (var y = 0; y < height; ++y)
            {
                //取置信度
                //从[1, 1, 80, 80]中取出第三、第四维：[80，80]
                //从[80，80]中取第二维数据[80]
                var scoresData = Enumerable.Range(0, height).Select(row => scores.At<float>(0, 0, y, row)).ToArray();

                //取位置
                //从[1, 5, 80, 80]中取出第三、第四维：[80，80] 以此排列
                //从[80，80]中取第二维数据[80]
                var x0Data = Enumerable.Range(0, height).Select(row => geometry.At<float>(0, 0, y, row)).ToArray();
                var x1Data = Enumerable.Range(0, height).Select(row => geometry.At<float>(0, 1, y, row)).ToArray();
                var x2Data = Enumerable.Range(0, height).Select(row => geometry.At<float>(0, 2, y, row)).ToArray();
                var x3Data = Enumerable.Range(0, height).Select(row => geometry.At<float>(0, 3, y, row)).ToArray();
                var anglesData = Enumerable.Range(0, height).Select(row => geometry.At<float>(0, 4, y, row)).ToArray();


                for (var x = 0; x < width; ++x)
                {
                    //从[80]中取出每一个得分
                    var score = scoresData[x];
                    if (score >= _config.ConfThreshold)       //大于设置的置信度得分则获取该对于的位置
                    {
                        // 解码操作：
                        // Multiple by 4 because feature maps are 4 time less than input image.
                        float offsetX = x * 4.0f;
                        float offsetY = y * 4.0f;
                        float angle = anglesData[x];

                        float cosA = (float)Math.Cos(angle);
                        float sinA = (float)Math.Sin(angle);
                        float x0 = x0Data[x];
                        float x1 = x1Data[x];
                        float x2 = x2Data[x];
                        float x3 = x3Data[x];

                        var h = x0 + x2;
                        var w = x1 + x3;


                        var value1 = offsetX + cosA * x1 + sinA * x2;
                        var value2 = offsetY - sinA * x1 + cosA * x2;
                        Point2f offset = new Point2f(value1, value2);
                        Point2f p1 = new Point2f(-sinA * h, -cosA * h) + offset;
                        Point2f p3 = new Point2f(-cosA * w, sinA * w) + offset;
                        RotatedRect r = new RotatedRect(new Point2f(0.5f * (p1.X + p3.X), 0.5f * (p1.Y + p3.Y)), new Size2f(w, h), (float)(-angle * 180.0f / Math.PI));

                        confidences.Add(score);
                        boxes.Add(r);
                    }
                }
            }
        }

        /// <summary>
        /// 将结果在图像上画出
        /// </summary>
        /// <param name="image"></param>
        /// <param name="vertices"> 位于图像中的位置</param>
        /// <param name="text"> 识别结果</param>
        private void Draw(Mat image, Point2f[] vertices, string text)
        {
            //区域
            for (int j = 0; j < 4; ++j)
            {
                Cv2.Line(image, (int)vertices[j].X, (int)vertices[j].Y, (int)vertices[(j + 1) % 4].X, (int)vertices[(j + 1) % 4].Y, new Scalar(0, 255, 0), 3);
            }

            //字符
            Cv2.PutText(image, text, (Point)vertices[1], HersheyFonts.HersheyTriplex, 1, Scalar.Red);
            Cv2.ImShow("图片展示：", image);

            //Cv2.ImWrite("C:/Users/tyy/Desktop/pass.bmp", image);
        }

        /// <summary>
        /// 旋转图像并剪切字符区域
        /// </summary>
        /// <param name="image"></param>
        /// <param name="vertices"></param>
        /// <param name="crops"></param>
        private void GetCropImage(Mat image, Point2f[] vertices, out Mat crops)
        {
            OpenCvSharp.Size outputSize = new OpenCvSharp.Size(100, 32);

            Point2f[] dsts = new Point2f[4];
            dsts[0] = new Point2f(0, outputSize.Height - 1);
            dsts[1] = new Point2f(0, 0);
            dsts[2] = new Point2f(outputSize.Width - 1, 0);
            dsts[3] = new Point2f(outputSize.Width - 1, outputSize.Height - 1);

            //计算旋转矩阵
            Mat rotationMatrix = Cv2.GetPerspectiveTransform(vertices, dsts);

            //旋转图像
            Mat rotatImag = new Mat();
            Cv2.WarpPerspective(image, rotatImag,rotationMatrix,outputSize);

            //
            Cv2.CvtColor(rotatImag, rotatImag, ColorConversionCodes.BGR2BGRA);

            crops = rotatImag;
        }

        /// <summary>
        /// 检测流程
        /// </summary>
        /// <param name="img"></param>
        /// <returns></returns>
        private OcrResult[] Process(Mat img)
        {
            Mat blob = DetectoterPretreatment(img);

            //推理
            _model.DetectoterNet.SetInput(blob);
            _model.DetectoterNet.Forward(_model.OutputBlobs, _model.OutputBlobNames);

            return Postprocess(img, _model.OutputBlobs);
        }

        /// <summary>
        /// 模型初始化
        /// </summary>
        private void InitDetectorModel()
        {
            _model.DetectoterNet = InitializeDetectoterModel(_config.DetectorModelPath);

            // 模型存在两个输出：1、字符位置；2、置信度分数
            // These are given by the layers :
            //   feature_fusion/concat_3
            //   feature_fusion/Conv_7/Sigmoid
            _model.OutputBlobNames = new string[] { "feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3" };
            _model.OutputBlobs = _model.OutputBlobNames.Select(_ => new Mat()).ToArray();
        }
        #endregion


        #region 识别模型
        /// <summary>
        ///  文本识别模型模型图片预处理
        /// </summary>
        /// <param Mat img</param>
        /// <returns></returns>
        private Mat RecognitionPretreatment(Mat img)
        {
            double scale = 1.0 / 127.5;
            Scalar scalar = new Scalar(127.5, 127.5, 127.5);

            Mat blob = CvDnn.BlobFromImage(img, scale, new Size(), scalar);

            return blob;
        }

        /// <summary>
        /// 初始化识别模型 打开pd模型
        /// </summary>
        private Net InitializeRecognitionModel(string pathModel)
        {
            if (!File.Exists(pathModel))
                throw new FileNotFoundException("The file model has not found", pathModel);

            Net net = CvDnn.ReadNetFromOnnx(pathModel);
            if (net == null || net.Empty())
                throw new Exception("The model has not yet initialized or is empty.");

            //读入模型和设置
            net.SetPreferableBackend(Backend.OPENCV);     // 3:DNN_BACKEND_OPENCV 
            net.SetPreferableTarget(Target.CPU);          //dnn target cpu

            return net;
        }

        /// <summary>
        /// 读入标签
        /// </summary>
        /// <param name="pathLabels"></param>
        /// <returns></returns>
        private string ReadLabels(string pathLabels)
        {
            if (!File.Exists(pathLabels))
                throw new FileNotFoundException("The file of labels not foud", pathLabels);

            string classNames = File.ReadAllText(pathLabels);

            return classNames;
        }

        /// <summary>
        /// 检测的后处理
        /// </summary>
        /// <param name="results"></param>
        /// <returns></returns>
        private string Postprocess(Mat scores)
        {
            //resharp from[N,1,37] to [N,37]
            Mat scoresMat = scores.Reshape(1, scores.Size(0));

            List<char> elements = new List<char>(scores.Size(0));

            for(int i =0; i <scoresMat.Rows; i++)
            {
                //从每一个[37]中获取最大置信度对于的序列号 
                Point max;
                Cv2.MinMaxLoc(scoresMat.Row(i), out _, out _, out _, out max);  

                if(max.X >0)
                {
                    elements.Add(_config.Alphabet[max.X - 1]);
                }
                else
                {
                    //对于实际生活中字符的间隔不定问题，在CTC模型中将空的区域当成‘-’
                    elements.Add('_');
                }
            }

            //将获取的字符编码转变为对于的ascall码
            string text = "";
            if (elements.Count > 0 && elements[0] != '_')
            {
                text += elements[0];
            }

            for(int j=1; j<elements.Count;++j)
            {
                if(j>0 && elements[j]!='_' && elements[j-1]!=elements[j])
                {
                    text += elements[j];
                }
            }

            return text;
        }

        /// <summary>
        /// 识别流程
        /// </summary>
        /// <param name="img"></param>
        /// <returns></returns>
        private string RecognitionProcess(Mat img)
        {
            Mat blob = RecognitionPretreatment(img);

            _model.RecognitionNet.SetInput(blob);
            var prob = _model.RecognitionNet.Forward();

            return Postprocess(prob);
        }

        /// <summary>
        /// 模型初始化
        /// </summary>
        private void InitRecognitionModel()
        {
            _model.RecognitionNet = InitializeRecognitionModel(_config.RecognitionModelPath);
        }
        #endregion

        #region demo
        private void Demo()
        {
            string dir = System.IO.Directory.GetCurrentDirectory();
            string dectecteoPath = System.IO.Path.Combine(dir, "frozen_east_text_detection.pb");
            string recognitionPath = System.IO.Path.Combine(dir, "CRNN_VGG_BiLSTM_CTC.onnx");
            string labelFile = System.IO.Path.Combine(dir, "Alphabet.txt");
            string imgpath = System.IO.Path.Combine(dir, "1.bmp");

            //初始化
            OCR_CRNNDetect ocr = new OCR_CRNNDetect(dectecteoPath, recognitionPath, labelFile);


            //检测
            OcrResult[] results = ocr.Detect(imgpath);

            foreach (var result in results)
            {
                Console.WriteLine("字符，区域坐标[第一点(x1,y1)、第二点(x2,y2)、第三点(x3,y3)、第四点(x4,y4)]");
                Console.WriteLine(result.Test + "  " + "[" +
                     "(" + result.Vertices[0].X.ToString() + "," + result.Vertices[0].Y.ToString() + ")" +
                     "(" + result.Vertices[1].X.ToString() + "," + result.Vertices[1].Y.ToString() + ")" +
                     "(" + result.Vertices[2].X.ToString() + "," + result.Vertices[2].Y.ToString() + ")" +
                     "(" + result.Vertices[3].X.ToString() + "," + result.Vertices[3].Y.ToString() + ")" + "]");
            }
        }
        #endregion
    }
}
