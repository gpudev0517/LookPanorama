#version 430
layout(location = 0) in vec2 texc;
layout(location = 0) out vec4 oColor;

uniform sampler2D colorTextures[8];
uniform sampler2D weightTextures[8];
uniform int viewCnt;
uniform int currentIndex;
uniform int eyeIndex;
uniform int paintMode;
uniform int eyeMode;

vec3 viewColorSet[8];

vec4 getCameraColor(int i, vec2 uv)
{
	return vec4(texture2D(colorTextures[i], uv).xyz, texture2D(weightTextures[i], uv).r);
}

void main(void)
{
	if(paintMode == 1 ){ //Paint Mode
		vec3 viewActiveColor = vec3(0.0f, 1.0f, 0.0f);
		vec3 viewOtherColor = vec3(0.5f, 0.5f, 0.5f);
		
		vec2 uv = vec2(texc.x, texc.y);
		float ws = 0.0f;
		vec3 csum = vec3(0.0f, 0.0f, 0.0f);
		for( int i = 0; i < viewCnt; i++ )
		{
			bool beyeIndex = bool(eyeIndex);
			vec3 viewColor ;
			if(eyeMode == 3){
				viewColor = (i == currentIndex) ? viewActiveColor : viewOtherColor;
			}
			else if (eyeMode == 4) {
				//!Mirror
				viewColor = (i == currentIndex) ? viewActiveColor : viewOtherColor;
			}
			else{
				bool isRightEye = (eyeMode == 2) ? true : false;
				viewColor = (beyeIndex == isRightEye && i == currentIndex) ? viewActiveColor : viewOtherColor;
			}
			vec4 src = getCameraColor(i, uv);
			vec3 csrc;
			if(i == currentIndex){
				if(eyeMode == 3)
					csrc = 0.2 * src.rgb + viewColor * src.a;
				else if (eyeMode == 4) {
					//!Mirror
					csrc = 0.2 * src.rgb + viewColor * src.a;
				}
				else{
					bool isRightEye = (eyeMode == 2) ? true : false;
					if(beyeIndex == isRightEye){
						csrc = 0.2 * src.rgb + viewColor * src.a;
					}
					else{
						csrc = src.rgb * viewColor * src.a;
					}
				}
			}
			else
				csrc = src.rgb * viewColor * src.a;

			csum = csum + csrc;
			ws = ws + src.a;
		}

	    if(ws == 0.0f)
	       csum = vec3(0.0f, 0.0f, 0.0f);

	    oColor = vec4(csum, 1.0f);
	}
	else if(paintMode == 2 ){ //View Mode

		vec2 uv = vec2(texc.x, texc.y);
		float ws = 0.0f;
		vec3 csum = vec3(0.0f, 0.0f, 0.0f);
		for( int i = 0; i < viewCnt; i++ )
		{
			vec4 src = getCameraColor(i, uv);
			csum = csum + src.rgb * src.a;
			ws = ws + src.a;
		}
		
	    if(ws == 0.0f)
	       csum = vec3(0.0f, 0.0f, 0.0f);
	    else
	       csum = csum / ws;
	    
	    oColor = vec4(csum, ws);
	}
	else if(paintMode == 3){  //Overlap Mode

		vec2 uv = vec2(texc.x, texc.y);
		float ws = 0.0f;
		vec3 csum = vec3(0.0f, 0.0f, 0.0f);
		int overlapIdx = 0;
		for( int i = 0; i < viewCnt; i++ )
		{
			vec4 src = getCameraColor(i, uv);

			if(src.a != 0){
				overlapIdx = overlapIdx + 1;
			}

			csum = csum + src.rgb * src.a;
			ws = ws + src.a;
		}
		
	    if(ws == 0.0f)
	       csum = vec3(0.0f, 0.0f, 0.0f);
	    else
	       csum = csum / ws;


	    if(overlapIdx > 1){
	    	csum = csum * 0.5;
	    }
	    
	    oColor = vec4(csum, ws);
	}
}