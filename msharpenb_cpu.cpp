/*
**						MSharpena Filter AVX2 and CUDA ver 0.0.2
**
**	The MIT License
**
**	Copyright (c) 2012-2014 miya <@sundola8x>
**
**	Permission is hereby granted, free of charge, to any person obtaining a copy of
**	this software and associated documentation files (the "Software"), to deal in
**	the Software without restriction, including without limitation the rights to
**	use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
**	of the Software, and to permit persons to whom the Software is furnished to do
**	so, subject to the following conditions:
**
**	The above copyright notice and this permission notice shall be included in all
**	copies or substantial portions of the Software.
**
**	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
**	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
**	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
**	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
**	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
**	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
**	SOFTWARE.
**
**	Thanks to the original plugin author: https://github.com/tp7/msharpen
*/
#include "stdafx.h"
#include "msharpenb.h"
#include "msharpenb_cpu.h"

typedef struct tagMSHARPEN_CORE_PARAM
{
	// ���L�������̐ݒ�
	int thread_step;
	// - �u���[��Ɨp
	PIXEL_YC *wblurx;
	// - �u���[��Ɨp2
	PIXEL_YC *wblury;
	// - �}�X�N��Ɨp
	bool     *wmask;
	//2014.09.01 ��SIMD�����Ɏ���
	// - �u���[��Ɨp���s�b�`
	int		wpitch;		
	//2014.09.01 ��SIMD�����Ɏ���

} MSHARPEN_CORE_PARAM, *LPMSHARPEN_CORE_PARAM;

#pragma region non simd filter

//---------------------------------------------------------------------
//		�C�����C���֐�
//---------------------------------------------------------------------
// 臒l�`�F�b�N
inline BOOL is_threshold_over(_In_ const PIXEL_YC yca, _In_ const PIXEL_YC ycb) {
	if (abs(yca.y - ycb.y) >= param.threshold) return TRUE;
	if (abs(yca.cb - ycb.cb) >= param.threshold) return TRUE;
	if (abs(yca.cr - ycb.cr) >= param.threshold) return TRUE;
	return FALSE;
}

//---------------------------------------------------------------------
//		�t�B���^�����֐� non simdi
//---------------------------------------------------------------------
void msharpen_core( 
	_In_ int thread_id,
	_In_ int thread_num,
	_In_ void *param1,
	_In_ void *param2)
{
	//========================================
	// �`�F�b�N
	//========================================
	FILTER_PROC_INFO *fpip	= (FILTER_PROC_INFO *)param2;
	assert(fpip != NULL);
	assert(fpip->ycp_edit != NULL);
	assert(fpip->ycp_temp != NULL);

	// ========================================
	// �}���`�X���b�h�ł̏�������
	// ========================================
	int y_start = ( fpip->h * thread_id     ) / thread_num;
	int y_end   = ( fpip->h * (thread_id+1) ) / thread_num;

	//========================================
	// ������
	//========================================
	int max_w  = fpip->max_w;						// �摜�̈�̃T�C�Y ����
	// �u���[�p��ƃ�����
//	int wpitch = (fpip->w + 7) & (~7);
	//PIXEL_YC *wblurx = new PIXEL_YC[(y_end - y_start + BLURX_MARGIN_Y * 2) * wpitch];
	//PIXEL_YC *wblury = new PIXEL_YC[(y_end - y_start + BLURY_MARGIN_Y * 2) * wpitch];
	//// �}�[�L���O�p��ƃ������̊m��
	//bool      *wmask = new bool    [(y_end - y_start                     ) * wpitch];
	MSHARPEN_CORE_PARAM* core_param = (MSHARPEN_CORE_PARAM*)param1;
	int wpitch = core_param->wpitch;
	PIXEL_YC *wblurx = core_param->wblurx + core_param->thread_step * thread_id * wpitch;
	PIXEL_YC *wblury = core_param->wblury + core_param->thread_step * thread_id * wpitch;
	// �}�[�L���O�p��ƃ������̊m��
	bool      *wmask = core_param->wmask  + core_param->thread_step * thread_id * wpitch;
	
	//========================================
	// �u���[ ��
	//========================================
	int y_start1 = y_start - BLURX_MARGIN_Y;	// �c�u���[�}�[�W�� 
	int y_end1   = y_end   + BLURX_MARGIN_Y;	// �c�u���[�}�[�W�� 
	if (y_start1 <  0      ) y_start1 = 0;
	if (y_end1   >= fpip->h) y_end1   = fpip->h;
	for (int y = y_start1 ; y < y_end1 ; y++ ) {
		// �����Ώۂ̃|�C���^�̌v�Z
		PIXEL_YC* ycp     = fpip->ycp_edit +  y * max_w;		// �摜�f�[�^
		PIXEL_YC* wblurxp = wblurx         + (y - y_start + BLURX_MARGIN_Y) * wpitch;
		for (int x = 0; x < fpip->w; x++ ) {
			if (x <= 1 || x >= fpip->w - 2 || y <= 1 || y >= fpip->h - 2) {
				//�t�B���^��������Ȃ��̈� ���͈̔̓`�F�b�N�͏��O���Ă��悢
				*wblurxp = *ycp;
			} else {
				//���u���[
				wblurxp->y  = (short)((22444 * (int)ycp[0].y  + 21546 * (int)ycp[-1].y  + 21546 * (int)ycp[+1].y ) >> 16);
				wblurxp->cb = (short)((22444 * (int)ycp[0].cb + 21546 * (int)ycp[-1].cb + 21546 * (int)ycp[+1].cb) >> 16);
				wblurxp->cr = (short)((22444 * (int)ycp[0].cr + 21546 * (int)ycp[-1].cr + 21546 * (int)ycp[+1].cr) >> 16);
			}
			ycp++;
			wblurxp++;
		}
	}

	//========================================
	// �u���[ �c
	//========================================
	y_start1 = y_start - BLURY_MARGIN_Y;	// �c�u���[�}�[�W�� 
	y_end1   = y_end   + BLURY_MARGIN_Y;	// �c�u���[�}�[�W�� 
	if (y_start1 <  0      ) y_start1 = 0;
	if (y_end1   >= fpip->h) y_end1   = fpip->h;
	for (int y = y_start1 ; y < y_end1 ; y++ ) {
		// �����Ώۂ̃|�C���^�̌v�Z
		PIXEL_YC* wblurxp   = wblurx + (y - y_start + BLURX_MARGIN_Y) * wpitch;
		PIXEL_YC* wbluryp   = wblury + (y - y_start + BLURY_MARGIN_Y) * wpitch;
		for (int x = 0; x < fpip->w; x++ ) {
			if (x <= 1 || x >= fpip->w - 2 || y <= 1 || y >= fpip->h - 2) {
				//�t�B���^��������Ȃ��̈� �c�͈̔̓`�F�b�N�͏��O���Ă��悢
				*wbluryp = *wblurxp;
			} else {
				//���u���[
				wbluryp->y  = (short)((22444 * (int)wblurxp[0].y  + 21546 * (int)wblurxp[-fpip->w].y  + 21546 * (int)wblurxp[+fpip->w].y ) >> 16);
				wbluryp->cb = (short)((22444 * (int)wblurxp[0].cb + 21546 * (int)wblurxp[-fpip->w].cb + 21546 * (int)wblurxp[+fpip->w].cb) >> 16);
				wbluryp->cr = (short)((22444 * (int)wblurxp[0].cr + 21546 * (int)wblurxp[-fpip->w].cr + 21546 * (int)wblurxp[+fpip->w].cr) >> 16);
			}
			wblurxp++;
			wbluryp++;
		}
	}

	//========================================
	// �}�[�L���O����
	//========================================
	for (int y = y_start ; y < y_end ; y++ ) {
		// �����Ώۂ̃|�C���^�̌v�Z
		PIXEL_YC *wbluryp = wblury + (y - y_start + BLURY_MARGIN_Y) * wpitch;	//�u���[
		bool     *wmaskp  = wmask  + (y - y_start                 ) * wpitch;	//�}�X�N
		for (int x = 0; x < fpip->w; x++ ) {
			if (x <= 1 || x >= fpip->w - 2 || y <= 1 || y >= fpip->h - 2) {
				//�t�B���^��������Ȃ��̈�
				*wmaskp = false;
			} else {
				// �΂߂��r(����,����)
				if        (is_threshold_over(wbluryp[-fpip->w-1], wbluryp[0])) {
					*wmaskp = true;
				// �΂߂��r(����,����)
				} else if (is_threshold_over(wbluryp[+fpip->w-1], wbluryp[0])) {
					*wmaskp = true;
				} else if (param.is_high_quality) {
					// �c���r(��,����)
					if        (is_threshold_over(wbluryp[-fpip->w  ], wbluryp[0])) {
						*wmaskp = true;
					// �����r(��,����)
					} else if (is_threshold_over(wbluryp[        -1], wbluryp[0])) {
						*wmaskp = true;
					// �c���r(��,����)
					} else if (is_threshold_over(wbluryp[+fpip->w  ], wbluryp[0])) {
						*wmaskp = true;
					// �΂ߔ�r(�E��,����)
					} else if (is_threshold_over(wbluryp[+fpip->w+1], wbluryp[0])) {
						*wmaskp = true;
					// �����r(�E,����)
					} else if (is_threshold_over(wbluryp[        +1], wbluryp[0])) {
						*wmaskp = true;
					// �΂ߔ�r(�E��,����)
					} else if (is_threshold_over(wbluryp[-fpip->w+1], wbluryp[0])) {
						*wmaskp = true;
					}
				}
			}
			wmaskp++;
			wbluryp++;
		}
	}

	//========================================
	// �}�X�N���ʂ̕ԋp
	//========================================
	if (param.is_mask) {
		for (int y = y_start; y < y_end; y++ ) {
			// �����Ώۂ̃|�C���^�̌v�Z
			PIXEL_YC *ycp_temp = fpip->ycp_temp +  y            * max_w;	// �e���|�����̈�
			bool     *wmaskp   = wmask          + (y - y_start) * wpitch;	//�}�X�N

			// ���������[�v
			for (int x = 0; x < fpip->w; x++ ) {
				// �F���̓t�B���^�̑ΏۊO
				ycp_temp->cb = 0;
				ycp_temp->cr = 0;
				// �P�x�ɑ΂��鏈��
				if (*wmaskp == true) {
					ycp_temp->y = 3072;
				} else {
					ycp_temp->y = 256;
				}
				//
				ycp_temp++;
				wmaskp++;
			}
		}
		return;
	}

	//========================================
	// ���ʂ̓K�p
	//========================================
	for (int y = y_start; y < y_end; y++ ) {
		// �����Ώۂ̃|�C���^�̌v�Z
		PIXEL_YC *ycp      = fpip->ycp_edit + y * max_w;						// �摜�f�[�^
		PIXEL_YC *ycp_temp = fpip->ycp_temp + y * max_w;						// �e���|�����̈�
		PIXEL_YC *wbluryp  = wblury + (y - y_start + BLURY_MARGIN_Y) * wpitch;	//�u���[
		bool     *wmaskp   = wmask  + (y - y_start                 ) * wpitch;	//�}�X�N

		// ���������[�v
		for (int x = 0; x < fpip->w; x++ ) {
			// �F���̓t�B���^�̑ΏۊO
			ycp_temp->cb = ycp->cb;
			ycp_temp->cr = ycp->cr;

			// �P�x�ɑ΂��鏈��
			if (*wmaskp == true) {
				int y = max(min(ycp->y * 4 - wbluryp->y * 3, 4096), 0);
				ycp_temp->y = (param.strength * y + param.invstrength * ycp->y) >> 8;
			} else {
				ycp_temp->y = ycp->y;
			}

			//
			ycp++;
			ycp_temp++;
			wbluryp++;
			wmaskp++;
		}
	}

	//delete[] wblurx;
	//delete[] wblury;
	//delete[] wmask;
}

//---------------------------------------------------------------------
//		�t�B���^�����֐�
//---------------------------------------------------------------------
BOOL cpu_filter_func( 
	_In_ FILTER* fp,
	_In_ FILTER_PROC_INFO* fpip)
{
	//========================================
	// �`�F�b�N
	//========================================
	assert(fp != NULL);
	assert(fp->track != NULL);
	assert(fp->check != NULL);
	assert(fpip != NULL);
	assert(fpip->ycp_edit != NULL);
	assert(fpip->ycp_temp != NULL);

	//========================================
	// ���s�X���b�h�����擾
	//========================================
	int thread_num;
	fp->exfunc->exec_multi_thread_func(msharpen_get_thread_num,(void *)&thread_num, NULL);

	//========================================
	// ���[�N�������̊m��
	//========================================
	MSHARPEN_CORE_PARAM param;
//	param.thread_step = fpip->h / thread_num + BLURX_MARGIN_Y * 2;
	param.thread_step =
		(fpip->h / thread_num) + (fpip->h % thread_num != 0 ? 1 : 0) +
		BLURX_MARGIN_Y * 2;
	//	int wpitch = (fpip->w + 7) & (~7);
	param.wpitch = (fpip->w + 31) & (~31);			//2014.08.31 AVX2�Ɍ����ĕ␳256bit=32byte
	param.wblurx = new PIXEL_YC[param.thread_step * thread_num * param.wpitch];
	param.wblury = new PIXEL_YC[param.thread_step * thread_num * param.wpitch];
	param.wmask  = new bool    [param.thread_step * thread_num * param.wpitch];

	//========================================
	// �t�B���^�[�̎��s
	//========================================
	fp->exfunc->exec_multi_thread_func(msharpen_core, (void *)&param, (void *)fpip);

	//========================================
	// ���[�N�������̌�n��
	//========================================
	delete[] param.wblurx;
	delete[] param.wblury;
	delete[] param.wmask;

	return TRUE;
}

#pragma endregion
