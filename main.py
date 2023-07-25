import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from streamlit import components

model = load_model('mymodel.h5')

st.title("DMD Scooter Classification App!")

st.image("https://www.intelligenttransport.com/wp-content/uploads/Lime-3.jpg", use_column_width=True)

file = st.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg"])




def predict_label(img):
    img = img.resize((150, 300))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return np.argmax(prediction)

def draw_prediction_on_image(image, prediction_text):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 30)  # You might need to provide full path to arial.ttf or any other font file
    text_position = (10, 10)  # You may need to adjust these values
    text_color = (0, 255, 0)  # RGB color, adjust as needed
    draw.text(text_position, prediction_text, fill=text_color, font=font)
    return image

# Animation HTML
success_animation = """
<div class="success-checkmark">
  <div class="check-icon">
    <span class="icon-line line-tip"></span>
    <span class="icon-line line-long"></span>
    <div class="icon-circle"></div>
    <div class="icon-fix"></div>
  </div>
</div>

<style>
/**
 * Extracted from: SweetAlert
 * Modified by: Istiak Tridip
 */
.success-checkmark {
    width: 80px;
    height: 115px;
    margin: 0 auto;
    
    .check-icon {
        width: 80px;
        height: 80px;
        position: relative;
        border-radius: 50%;
        box-sizing: content-box;
        border: 4px solid #4CAF50;
        
        &::before {
            top: 3px;
            left: -2px;
            width: 30px;
            transform-origin: 100% 50%;
            border-radius: 100px 0 0 100px;
        }
        
        &::after {
            top: 0;
            left: 30px;
            width: 60px;
            transform-origin: 0 50%;
            border-radius: 0 100px 100px 0;
            animation: rotate-circle 4.25s ease-in;
        }
        
        &::before, &::after {
            content: '';
            height: 100px;
            position: absolute;
            background: #FFFFFF;
            transform: rotate(-45deg);
        }
        
        .icon-line {
            height: 5px;
            background-color: #4CAF50;
            display: block;
            border-radius: 2px;
            position: absolute;
            z-index: 10;
            
            &.line-tip {
                top: 46px;
                left: 14px;
                width: 25px;
                transform: rotate(45deg);
                animation: icon-line-tip 0.75s;
            }
            
            &.line-long {
                top: 38px;
                right: 8px;
                width: 47px;
                transform: rotate(-45deg);
                animation: icon-line-long 0.75s;
            }
        }
        
        .icon-circle {
            top: -4px;
            left: -4px;
            z-index: 10;
            width: 80px;
            height: 80px;
            border-radius: 50%;
            position: absolute;
            box-sizing: content-box;
            border: 4px solid rgba(76, 175, 80, .5);
        }
        
        .icon-fix {
            top: 8px;
            width: 5px;
            left: 26px;
            z-index: 1;
            height: 85px;
            position: absolute;
            transform: rotate(-45deg);
            background-color: #FFFFFF;
        }
    }
}

@keyframes rotate-circle {
    0% {
        transform: rotate(-45deg);
    }
    5% {
        transform: rotate(-45deg);
    }
    12% {
        transform: rotate(-405deg);
    }
    100% {
        transform: rotate(-405deg);
    }
}

@keyframes icon-line-tip {
    0% {
        width: 0;
        left: 1px;
        top: 19px;
    }
    54% {
        width: 0;
        left: 1px;
        top: 19px;
    }
    70% {
        width: 50px;
        left: -8px;
        top: 37px;
    }
    84% {
        width: 17px;
        left: 21px;
        top: 48px;
    }
    100% {
        width: 25px;
        left: 14px;
        top: 45px;
    }
}

@keyframes icon-line-long {
    0% {
        width: 0;
        right: 46px;
        top: 54px;
    }
    65% {
        width: 0;
        right: 46px;
        top: 54px;
    }
    84% {
        width: 55px;
        right: 0px;
        top: 35px;
    }
    100% {
        width: 47px;
        right: 8px;
        top: 38px;
    }
}
</style>

<script>
$("button").click(function () {
  $(".check-icon").hide();
  setTimeout(function () {
    $(".check-icon").show();
  }, 10);
});
</script>
"""

failure_animation_orange = """
<div class="error-checkmark">
  <div class="check-icon">
    <span class="icon-line line-horizontal"></span>
    <span class="icon-line line-vertical"></span>
    <div class="icon-circle"></div>
    <div class="icon-fix"></div>
  </div>
</div>

<style>
.error-checkmark {
    width: 80px;
    height: 115px;
    margin: 0 auto;

    .check-icon {
        width: 80px;
        height: 80px;
        position: relative;
        border-radius: 50%;
        box-sizing: content-box;
        border: 4px solid orange;

        .icon-line {
            width: 50%;
            height: 4px;
            background-color: orange;
            position: absolute;
            top: 50%;
            left: 50%;
            transform-origin: center center;
            transform: translate(-50%, -50%);

            &.line-horizontal {
                transform: translate(-50%, -50%) rotate(45deg);
            }

            &.line-vertical {
                transform: translate(-50%, -50%) rotate(-45deg);
            }
        }

        .icon-circle {
            top: -4px;
            left: -4px;
            z-index: 10;
            width: 80px;
            height: 80px;
            border-radius: 50%;
            position: absolute;
            box-sizing: content-box;
            border: 4px solid rgba(255, 165, 0, .5);
        }

        .icon-fix {
            display: none;
        }
    }
}
</style>

<script>
$("button").click(function () {
  $(".check-icon").hide();
  setTimeout(function () {
    $(".check-icon").show();
  }, 10);
});
</script>
"""

failure_animation = """
<div class="error-checkmark">
  <div class="check-icon">
    <span class="icon-line line-horizontal"></span>
    <span class="icon-line line-vertical"></span>
    <div class="icon-circle"></div>
    <div class="icon-fix"></div>
  </div>
</div>

<style>
.error-checkmark {
    width: 80px;
    height: 115px;
    margin: 0 auto;

    .check-icon {
        width: 80px;
        height: 80px;
        position: relative;
        border-radius: 50%;
        box-sizing: content-box;
        border: 4px solid #f86;

        .icon-line {
            width: 50%;
            height: 4px;
            background-color: #f86;
            position: absolute;
            top: 50%;
            left: 50%;
            transform-origin: center center;
            transform: translate(-50%, -50%);

            &.line-horizontal {
                transform: translate(-50%, -50%) rotate(45deg);
            }

            &.line-vertical {
                transform: translate(-50%, -50%) rotate(-45deg);
            }
        }

        .icon-circle {
            top: -4px;
            left: -4px;
            z-index: 10;
            width: 80px;
            height: 80px;
            border-radius: 50%;
            position: absolute;
            box-sizing: content-box;
            border: 4px solid rgba(248, 86, 86, .5);
        }

        .icon-fix {
            display: none;
        }
    }
}
</style>

<script>
$("button").click(function () {
  $(".check-icon").hide();
  setTimeout(function () {
    $(".check-icon").show();
  }, 10);
});
</script>
"""


def get_class_name(prediction):
    if prediction == 0:
        return success_animation, "<div style='color:green; font-size:20px; font-weight:bold'>Your scooter is parked correctly. Have a good day!</div>"
    elif prediction == 1:
        return failure_animation, "<div style='color:red; font-size:20px; font-weight:bold'>Sorry but we can't close your trip... There is no scooter in you picture. Try to take another picture.</div>"
    elif prediction == 2:
        return failure_animation_orange, "<div style='color:orange; font-size:20px; font-weight:bold'>Your scooter is upside down! Please Park it properly.</div>"
    else:
        return failure_animation_orange, "<div style='color:orange; font-size:20px; font-weight:bold'>Your scooter is blocking the footpath! Please park it properly.</div>"


if file is not None:
    image = Image.open(file).convert("RGB")
    display_image = image.resize((300, 300))  # Resize the image for display only

    col1, col2 = st.columns(2)

    with col1:
        st.image(display_image, use_column_width=True)

    prediction = predict_label(image)  # Use original image for prediction
    animation, prediction_name = get_class_name(prediction)

    with col2:
        components.v1.html(animation, height=220)  # Add success animation
        st.markdown(prediction_name, unsafe_allow_html=True)

