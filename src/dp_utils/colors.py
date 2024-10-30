class Color:
    def __init__(self, red: float, green: float, blue: float):
        self.red = red
        self.green = green
        self.blue = blue
        
    @property
    def rgb(self):
        return self.red, self.green, self.blue
    
    @property
    def hex(self):
        # Convert each RGB component to a 0-255 integer, format as hex, and return as a string
        return "#{:02x}{:02x}{:02x}".format(
            int(self.red * 255),
            int(self.green * 255),
            int(self.blue * 255)
        )

    def rgba(self, alpha: float = 1.0):
        return self.red, self.green, self.blue, alpha

    def __call__(self, alpha: float = 1.0):
        if alpha == 1.0:
            return self.rgb
        else:
            return self.rgba(alpha)

BLUE    = Color(0.12156862745098039, 0.4666666666666667, 0.7058823529411765)
ORANGE  = Color(1.0, 0.4980392156862745, 0.054901960784313725)
GREEN   = Color(0.17254901960784313, 0.6274509803921569, 0.17254901960784313)
RED     = Color(0.8392156862745098, 0.15294117647058825, 0.1568627450980392)
PURPLE  = Color(0.5803921568627451, 0.403921568627451, 0.7411764705882353)
BROWN   = Color(0.5490196078431373, 0.33725490196078434, 0.29411764705882354)
PINK    = Color(0.8901960784313725, 0.4666666666666667, 0.7607843137254902)
GRAY    = Color(0.4980392156862745, 0.4980392156862745, 0.4980392156862745)
OLIVE   = Color(0.7372549019607844, 0.7411764705882353, 0.13333333333333333)
CYAN    = Color(0.09019607843137255, 0.7450980392156863, 0.8117647058823529)