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
/*
���͂������̂����ł��B�s���_������΃c�C�b�^�[@sundola8x�Ƀ_�C���N�g���b�Z�[�W�ŋX�������肢���܂��B
�t�B���^�̏������e�ɑ����������Ă��Ȃ��ꍇ���������܂��̂őS�Ă����������邱�Ƃ͂ł��܂���B
���̃v���O�����ɂ�鑹���͓����ł͕������˂܂��B���g�p�͎��ȐӔC�ɂčs���Ă��������B

MSharpen�t�B���^�ɂ���
    MSharpena��Aviutl�̂��߂ɍĎ��������AAviSynth�̔��ɃV���v���ȃG�b�W�����v���O�C���ł��B
    ���̃t�B���^�́A�Â�neuron2�̃v���O�C�����ڐA�������̂ł��B
    ���̃t�B���^�͔���p�u���[�摜�𐶐����ăG�b�W���o�������s���Ă��邽�߃m�C�Y�ɋ����ƕ]��������܂��B

���\�z���@�F
    Aviutl�z���̃t�H���_�ɉ��L�t�@�C�����R�s�[���Ă��������B
        msvcr120.dll
        msvcp120.dll
        cudart32_65.dll
        cuda_manager.dll ����DLL���V�����Ȃ��Ă��܂��B
    Aviutl�z���̃t�H���_��plugins�t�H���_�ɉ��L�t�@�C�����R�s�[���Ă��������B
        msharpenb.auf

�����(�e�X�g��)�F
    OS     : Windows7 Ultimate 64bit SP1
    Aviutl : 0.99l
    CPU    : Intel Core i7-4790K 4.0GHz
    Mem    : 16GB
    GPU    : NVIDIA Geforce GTX 780Ti
             NVIDIA��GPU�h���C�o�[�A340.62 (�Â��o�[�W�����ł����삷�邩������܂���B)
             GPU��NVIDIA Geforce GTX 560Ti �ȏ㐄���@����ȊO�ł������\������B�e�X�^�[��W

�ݒ�l�̐����F(�{�Ɛ��������p)
    臒l : (0-255�A�f�t�H���g:15)
        ��������G�b�W�������邽�߂Ɏg�p�����܂��B�l������������Ƌ�������ӏ��������܂��B
        �������������m�F����ɂ́u�}�X�N���[�h�v�Ƀ`�F�b�N�����Ă��������B
    ���� : (0-255�A�f�t�H���g:100)
        ��������G�b�W�ɓK�p������s���̋����B�l��傫������ƃG�b�W�̋����x�������܂��B
    ���i�����[�h : (�f�t�H���g:�I�t)
        �`�F�b�N�����ƃG�b�W�����o���菈���𑝂₵�Đ��x�����߂܂��B�኱�d���Ȃ�܂��B
    �}�X�N���[�h : (�f�t�H���g:�I�t)
        �`�F�b�N������Ƌ������������m�F���邱�Ƃ��ł��܂��B
    GPU(CUDA)���g�p���� : (�f�t�H���g:�I�t)
        �`�F�b�N������ƃt�B���^������GPU�ɂ�点�܂��B�`�F�b�N���O���ƃt�B���^������CPU�ɂ�点�܂��B
�@�@�@�@NVIDIA GTX 780Ti �ȏ��GPU�Ńp�t�H�[�}���X�����P���Ă��܂��B
    AVX2���g�p���� : (�f�t�H���g:�I�t)
�@�@�@�@�`�F�b�N������ƃt�B���^������AVX2���g�p���܂��B
�@�@�@�@���ӁI�I�IAVX2���g�p�ł��邩�`�F�b�N�����Ă��܂���̂Ŏ��ȐӔC�Ń`�F�b�N�����Ă��������B
�@�@�@�@�@�@�@�@�@�u GPU(CUDA)���g�p����v�Ƀ`�F�b�N�������Ă���ꍇ�͂��������D�悳��܂��B
�@�@�@�@AVX2���g�p�����ꍇ�̏������ʂ́A���̃t�B���^�Ǝ኱�덷�������܂��̂ł��������������B
    �h���b�v�t���[���`�F�b�N(�e�X�g) : (�f�t�H���g:�I�t)
        �`�F�b�N������Ǝ����t�B�[���h�V�t�g���쎞�̃h���b�v�t���[���ŁA
        �t�B���^�̏������X�L�b�v���邽�ߖ��ʂȏ��������点�܂��B
        �������A�e�X�g�����Ȃ̂Ŏ����t�B�[���h�V�t�g�̎��h���b�v�t���[����
        ���肪�قȂ邱�Ƃ�����܂��B�I�t�����ł��B

�{�ƃT�C�gURL https://github.com/tp7/msharpen

�X�V����
  Ver.0.0.1 -- test version
  Ver.0.0.2 -- �n���ɍ�����
*/
#include "stdafx.h"
#include "msharpenb.h"
#include <msharpenbLibCUDA20_cuda.h>
#include <msharpenbLibCUDA35_cuda.h>
#include "msharpenb_cpu.h"
#include <afs_check.h>

//---------------------------------------------------------------------
//		�֐��v���g�^�C�v
//---------------------------------------------------------------------
extern BOOL func_proc(_In_ FILTER* fp, _In_ FILTER_PROC_INFO* fpip);
extern BOOL func_init(_In_ FILTER* fp);
extern BOOL func_exit(_In_ FILTER* fp);
extern BOOL func_update(_In_ FILTER *fp, _In_ int status );
//---------------------------------------------------------------------
//		�t�B���^�\���̒�`
//---------------------------------------------------------------------
// �^�C�g��
static const LPSTR lpszFilterTilte = "msharpen +beta";
static const LPSTR lpszFilterCOMM = "msharpen +beta Ver.0.0.2 MT @sundola8x";
// ��`
static int TRACK_N = 2;														//	�g���b�N�o�[�̐�
static TCHAR* track_name[] ={"臒l","����"	};								//	�g���b�N�o�[�̖��O
static int track_default[] ={15,	100,	};								//	�g���b�N�o�[�̏����l
static int track_s[] =		{0,		0,		};								//	�g���b�N�o�[�̉����l
static int track_e[] =		{255,	255,	};								//	�g���b�N�o�[�̏���l
static int CHECK_N = 5;														//	�`�F�b�N�{�b�N�X�̐�
static TCHAR* check_name[] ={												//	�`�F�b�N�{�b�N�X�̖��O
	"���i�����[�h",
	"�}�X�N���[�h",
	"GPU(CUDA)���g�p����",
	"AVX2���g�p����",
	"�h���b�v�t���[���`�F�b�N(�e�X�g)" };
static int check_default[] ={												//	�`�F�b�N�{�b�N�X�̏����l (�l��0��1)
	FALSE,		//	���i�����[�h
	FALSE,		//	�}�X�N���[�h
	FALSE,		//	GPU(CUDA)���g�p����
	FALSE,		//	AVX2���g�p����
	FALSE };	//	�h���b�v�t���[���`�F�b�N(�e�X�g)

static FILTER_DLL filter = {
	FILTER_FLAG_EX_INFORMATION,
	0,0,
	lpszFilterTilte,
	TRACK_N,
	track_name,
	track_default,
	track_s,track_e,
	CHECK_N,
	check_name,
	check_default,
	func_proc,
	func_init,//
	func_exit,//
	func_update,//
	NULL,//
	NULL,NULL,
	NULL,
	NULL,
	lpszFilterCOMM,
	NULL,
	NULL,
};

//---------------------------------------------------------------------
//		�O���[�o��
//---------------------------------------------------------------------
// �ǂ����ɕۑ��ł���΂����̂���
MSHARPEN_PARAM param;
// �g�p����CPU�t�B���^�֐���ۑ�
BOOL(*use_cpu_filter_func)(_In_ FILTER* fp, _In_ FILTER_PROC_INFO* fpip);
// �g�p����GPU�t�B���^�֐���ۑ�
BOOL(*use_cuda_filter_func)(_In_ FILTER* fp, _In_ FILTER_PROC_INFO* fpip, _Inout_ MSHARPEN_PARAM* paramp);

#pragma region **** �t�B���^������ ****

//---------------------------------------------------------------------
// �t�B���^��b�f�[�^�̍\�z
//---------------------------------------------------------------------
inline BOOL __stdcall _init_param(
	_In_ const FILTER* fp)
{
	// ========================================
	// �t�B���^�̐ݒ���i�[
	// ========================================
	param.threshold = fp->track[TRACK_THRESHOLD] << 4;	// �t�B���^�\����->�g���b�N�o�[�̐ݒ�l[臒l]
	param.strength = fp->track[TRACK_STRENGTH];		// �t�B���^�\����->�g���b�N�o�[�̐ݒ�l[����]
	param.invstrength = 255 - fp->track[TRACK_STRENGTH];	// �t�B���^�\����->�g���b�N�o�[�̐ݒ�l[����]
	param.is_high_quality = fp->check[CHECK_HIGH_QUALITY];	// �t�B���^�\����->�`�F�b�N�{�b�N�X�̐ݒ�l[���i��]
	param.is_mask = fp->check[CHECK_MASK];			// �t�B���^�\����->�`�F�b�N�{�b�N�X�̐ݒ�l[�}�X�N]

	// ========================================
	// �g�p����CPU�t�B���^�֐���ۑ�
	// ========================================
	if ((param.is_avx2_use == TRUE && fp->check[CHECK_AVX2_USE] == FALSE) || 
		(param.is_avx2_use == FALSE && fp->check[CHECK_AVX2_USE] == TRUE) ||
		use_cpu_filter_func == NULL) {
		if (fp->check[CHECK_AVX2_USE] == TRUE) {
			if (get_availableSIMD() & AVX2 != 0) {
				DBG("using cpu_filter_avx2_func \n");
				use_cpu_filter_func = cpu_filter_avx2_func;
				param.is_avx2_use = TRUE;
			}
			else {
				DBG("using cpu_filter_func \n");
				use_cpu_filter_func = cpu_filter_func;
				param.is_avx2_use = FALSE;
				fp->check[CHECK_AVX2_USE] = FALSE;
				fp->exfunc->filter_window_update((void*)fp);
			}
		}
		else {
			DBG("using cpu_filter_func \n");
			use_cpu_filter_func = cpu_filter_func;
			param.is_avx2_use = FALSE;
		}
	}

	return  TRUE;
}

#pragma endregion

#pragma region **** �O���ďo���p ****

//---------------------------------------------------------------------
//		�t�B���^�\���̂̃|�C���^��n���֐�
//---------------------------------------------------------------------
EXTERN_C FILTER_DLL __declspec(dllexport) * __stdcall GetFilterTable( void )
{
	return &filter;
}

//---------------------------------------------------------------------
//		�t�B���^�������֐�
//---------------------------------------------------------------------
BOOL func_init( 
	_In_ FILTER* fp)
{
	//========================================
	// �`�F�b�N
	//========================================
	assert(fp != NULL);
	assert(fp->track != NULL);
	assert(fp->check != NULL);

	//========================================
	// �f�o�b�O ������
	//========================================
	DBG_INIT();

	// ========================================
	// �t�B���^��b�f�[�^�̍\�z
	// ========================================
	ZeroMemory(&param, sizeof(param));
	_init_param(fp);

	return TRUE;
}

//---------------------------------------------------------------------
//		�t�B���^��n���֐�
//---------------------------------------------------------------------
BOOL func_exit( 
	_In_ FILTER* fp)
{
	//========================================
	// �`�F�b�N
	//========================================
	assert(fp != NULL);
	assert(fp->track != NULL);
	assert(fp->check != NULL);

	//========================================
	// CUDA ��n��
	//========================================
	//cuda_manager_exit();

	//========================================
	// �f�o�b�O�p ��n��
	//========================================
	DBG_EXIT();

	return TRUE;
}

//---------------------------------------------------------------------
//		�t�B���^�X�V�֐�
//---------------------------------------------------------------------
BOOL func_update(
	_In_ FILTER *fp,
	_In_ int status)
{
	// ========================================
	// �t�B���^�̐ݒ肪�ς�邽�тɌĂяo����邽��
	// �ݒ�ɕύX���Ȃ��ꍇ�ɕω����Ȃ��f�[�^�������ōX�V����ƌ����I
	// ========================================
	DBG("update %8x\n", status);

	// ========================================
	// �t�B���^��b�f�[�^�̍\�z
	// ========================================
	// �S�Ă̐ݒ肪�ύX���ꂽ�ꍇ
	if (status == FILTER_UPDATE_STATUS_ALL) {
		DBG("_init_param \n");
		_init_param(fp);
	// �g���b�N�o�[�̐ݒ肪�ύX���ꂽ�ꍇ
	} else if (status & FILTER_UPDATE_STATUS_TRACK) {
		DBG("_init_param \n");
		_init_param(fp);
	// �`�F�b�N�{�b�N�X�̐ݒ肪�ύX���ꂽ�ꍇ
	} else if (status & FILTER_UPDATE_STATUS_CHECK) {
		DBG("_init_param \n");
		_init_param(fp);
	}

	return TRUE;
}

//---------------------------------------------------------------------
//		�t�B���^�����֐�
//---------------------------------------------------------------------
BOOL func_proc( 
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

	// ========================================
	// �h���b�v�t���[���`�F�b�N(�e�X�g)
	// ========================================
	if (fp->check[CHECK_SKIP_DROP_FRAME] && is_afs_drop_frame(fp, fpip)) {
		return TRUE;
	}

	//========================================
	// �t�B���^�[�̐؂�ւ�
	//========================================
	if (fp->check[CHECK_GPU_USE] == TRUE && 
		cuda_manager_init(fp, fpip) == TRUE) { 
        // �g�p�t�B���^��I��
		if (use_cuda_filter_func == NULL) {
			if (get_cuda_compute_capability_major() > 3 || 
				(get_cuda_compute_capability_major() == 3 && get_cuda_compute_capability_minor() >= 5)) {
				DBG("using cuda_filter_CUDA35_func \n");
				use_cuda_filter_func = cuda_filter_CUDA35_func;
			}
			else {
				DBG("using cuda_filter_CUDA20_func \n");
				use_cuda_filter_func = cuda_filter_CUDA20_func;
			}
		}

		if (use_cuda_filter_func(fp, fpip, &param) == FALSE)
			return FALSE;
	} else {
		if (use_cpu_filter_func(fp, fpip) == FALSE)
			return FALSE;
	}

	//========================================
	// �A�h���X����
	//========================================
	PIXEL_YC* temp = fpip->ycp_edit;
	fpip->ycp_edit = fpip->ycp_temp;
	fpip->ycp_temp = temp;

	return TRUE;
}

#pragma endregion
