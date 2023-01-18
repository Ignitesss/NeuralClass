using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;

namespace NeuralNetwork1
{
    internal class Settings
    {
        private int _border = 20;
        public int border
        {
            get
            {
                return _border;
            }
            set
            {
                if ((value > 0) && (value < height / 3))
                {
                    _border = value;
                    if (top > 2 * _border) top = 2 * _border;
                    if (left > 2 * _border) left = 2 * _border;
                }
            }
        }

        public int width = 640;
        public int height = 640;

        /// <summary>
        /// Размер сетки для сенсоров по горизонтали
        /// </summary>
        public int blocksCount = 10;

        /// <summary>
        /// Желаемый размер изображения до обработки
        /// </summary>
        public Size orignalDesiredSize = new Size(500, 500);
        /// <summary>
        /// Желаемый размер изображения после обработки
        /// </summary>
        public Size processedDesiredSize = new Size(300, 300);

        public int margin = 10;
        public int top = 40;
        public int left = 40;

        /// <summary>
        /// Второй этап обработки
        /// </summary>
        public bool processImg = false;

        /// <summary>
        /// Порог при отсечении по цвету 
        /// </summary>
        public byte threshold = 120;
        public float differenceLim = 0.15f;

        public void incTop() { if (top < 2 * _border) ++top; }
        public void decTop() { if (top > 0) --top; }
        public void incLeft() { if (left < 2 * _border) ++left; }
        public void decLeft() { if (left > 0) --left; }
    }

    internal class MagicEye
    {
        /// <summary>
        /// Обработанное изображение
        /// </summary>
        public Bitmap processed;
        /// <summary>
        /// Оригинальное изображение после обработки
        /// </summary>
        public Bitmap original;

        /// <summary>
        /// Класс настроек
        /// </summary>
        public Settings settings = new Settings();

        
        private BaseNetwork network;
        private DatasetProcessor dataset;
        public MagicEye(BaseNetwork network, DatasetProcessor dataset)
        {
            this.network = network;
            this.dataset = dataset;
        }

        public bool ProcessImage(Bitmap bitmap)
        {
          
            if (bitmap.Height > bitmap.Width)
                throw new Exception("Bro what");
            int side = bitmap.Height;


          
            Rectangle cropRect = new Rectangle(settings.border, settings.border, bitmap.Width - settings.border, bitmap.Height - settings.border);


            original = bitmap;

           
            AForge.Imaging.Filters.Crop cropFilter = new AForge.Imaging.Filters.Crop(cropRect);
            var uProcessed = cropFilter.Apply(AForge.Imaging.UnmanagedImage.FromManagedImage(original));
            AForge.Imaging.Filters.Grayscale grayFilter = new AForge.Imaging.Filters.Grayscale(0.2125, 0.7154, 0.0721);
            uProcessed = grayFilter.Apply(uProcessed);



           
            AForge.Imaging.Filters.ResizeBilinear scaleFilter = new AForge.Imaging.Filters.ResizeBilinear(settings.processedDesiredSize.Width, settings.processedDesiredSize.Height);
            uProcessed = scaleFilter.Apply(uProcessed);
 


            AForge.Imaging.Filters.BradleyLocalThresholding threshldFilter = new AForge.Imaging.Filters.BradleyLocalThresholding();
            threshldFilter.PixelBrightnessDifferenceLimit = settings.differenceLim;
            threshldFilter.ApplyInPlace(uProcessed);
            
            processed = uProcessed.ToManagedImage();
            
            if (settings.processImg)
            {
                currentType = processSample(processed);
            }

            return true;
        }
        
        public LetterType currentType { get; set; }

        /// <summary>
        /// Обработка одного сэмпла
        /// </summary>
        /// <param name="index"></param>
        private LetterType processSample(Bitmap bitmap)
        {
            Sample s = dataset.getSample(bitmap);
            return network.Predict(s);
        }

    }
}
