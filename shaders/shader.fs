#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

// texture sampler
uniform sampler2D texture1;
uniform vec4 color;
uniform bool randColor;
uniform vec4 background;

void main()
{
    vec4 randomColor = texture(texture1, TexCoord);
    if(randColor)
        FragColor = mix(background, randomColor, randomColor);
    else
        FragColor = mix(background, color, randomColor.a);

}
