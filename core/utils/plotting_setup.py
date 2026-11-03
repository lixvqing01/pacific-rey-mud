import logging
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def setup_matplotlib_chinese_font():
    """
    Sets up matplotlib to correctly display Chinese characters.
    It tries to use 'SimHei' or 'Microsoft YaHei' fonts.
    """

    preferred_fonts = ['SimHei', 'Microsoft YaHei']
    font_set = False
    for font_name in preferred_fonts:
        try:
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False


            logger.info(f"Matplotlib font set to {font_name} for Chinese characters.")
            font_set = True
            break
        except Exception as e:
            logger.warning(f"Failed to set {font_name}. Error: {e}")

    if not font_set:
        logger.warning(f"Could not set any of the preferred fonts: {preferred_fonts}.")
        logger.warning("Please ensure that a suitable Chinese font (e.g., 'SimHei', 'Microsoft YaHei') is installed on your system and accessible by Matplotlib.")
        logger.warning("You might need to clear Matplotlib's font cache after installing new fonts.")
        logger.warning("Consider adding a fallback to a generic sans-serif font if Chinese display is not critical or if specific fonts cannot be guaranteed.")


