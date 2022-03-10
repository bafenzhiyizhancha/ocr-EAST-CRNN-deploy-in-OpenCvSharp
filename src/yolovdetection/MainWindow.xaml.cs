using System;
using System.Windows;
using OCR_CRNN;

namespace yolovdetection
{
    /// <summary>
    /// MainWindow.xaml 的交互逻辑
    /// </summary>
    public partial class MainWindow : Window
    {
        private OCR_CRNNDetect ocr;
        private string dir;


        public MainWindow()
        {
            InitializeComponent();
        }

        //ocr
        private void Init3()
        {

            dir               = System.IO.Directory.GetCurrentDirectory();
            string weightpath = dir + "/weights/";
            
            string dectecteoPath    = System.IO.Path.Combine(weightpath, "frozen_east_text_detection.pb");
            string recognitionPath  = System.IO.Path.Combine(weightpath, "CRNN_VGG_BiLSTM_CTC.onnx");
            string labelFile        = System.IO.Path.Combine(weightpath, "Alphabet.txt");   

            ocr = new OCR_CRNNDetect(dectecteoPath, recognitionPath, labelFile);
           
        }

        //ocr
        private void Detect3()
        {
            string imgagepath   = dir + "/image/";
            string imgpath      = System.IO.Path.Combine(imgagepath, "1.bmp");
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

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            Detect3();
        }

        private void Button_Click_1(object sender, RoutedEventArgs e)
        {
            Init3();
        }
    }
}
