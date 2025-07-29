import 'package:figma_squircle/figma_squircle.dart';
import 'package:flutter/material.dart';
import "package:photos/ente_theme_data.dart";
import "package:photos/theme/ente_theme.dart";
import 'package:pro_image_editor/pro_image_editor.dart'; 

final filterList = [
  FilterModel(
    name: "Juno",
    filters: [
      ColorFilterAddons.rgbScale(1.01, 1.04, 1),
      ColorFilterAddons.saturation(0.3),
    ],
  ),
  FilterModel(
    name: 'Perpetua',
    filters: [
      ColorFilterAddons.rgbScale(1.05, 1.1, 1),
    ],
  ),
  FilterModel(
    name: 'Reyes',
    filters: [
      ColorFilterAddons.sepia(0.4),
      ColorFilterAddons.brightness(0.13),
      ColorFilterAddons.contrast(-.05),
    ],
  ),
  FilterModel(
    name: 'Aden',
    filters: [
      ColorFilterAddons.colorOverlay(228, 130, 225, 0.13),
      ColorFilterAddons.saturation(-0.2),
    ],
  ),
  FilterModel(
    name: "New preset",
    filters: [
      ColorFilterAddons.hue(-0.6),
      ColorFilterAddons.rgbScale(0.8, 1.0, 1.2),
      ColorFilterAddons.saturation(-0.8),
      ColorFilterAddons.contrast(-0.6),
    ],
  ),
  FilterModel(
    name: 'Amaro',
    filters: [
      ColorFilterAddons.saturation(0.3),
      ColorFilterAddons.brightness(0.15),
    ],
  ),
  FilterModel(
    name: 'Clarendon',
    filters: [
      ColorFilterAddons.brightness(.1),
      ColorFilterAddons.contrast(.1),
      ColorFilterAddons.saturation(.15),
    ],
  ),
  FilterModel(
    name: 'Brooklyn',
    filters: [
      ColorFilterAddons.colorOverlay(25, 240, 252, 0.05),
      ColorFilterAddons.sepia(0.3),
    ],
  ),
  FilterModel(
    name: 'Sierra',
    filters: [
      ColorFilterAddons.contrast(-0.15),
      ColorFilterAddons.saturation(0.1),
    ],
  ),
  FilterModel(
    name: 'Inkwell',
    filters: [
      ColorFilterAddons.contrast(0.2),
      ColorFilterAddons.grayscale(),
    ],
  ),
];

class ImageEditorFilterBar extends StatefulWidget {
  const ImageEditorFilterBar({
    required this.filterModel,
    required this.isSelected,
    required this.onSelectFilter,
    required this.editorImage,
    this.filterKey,
    super.key,
  });

  final FilterModel filterModel;
  final bool isSelected;
  final VoidCallback onSelectFilter;
  final Widget editorImage;
  final Key? filterKey;

  @override
  State<ImageEditorFilterBar> createState() => _ImageEditorFilterBarState();
}

class _ImageEditorFilterBarState extends State<ImageEditorFilterBar> {
  @override
  Widget build(BuildContext context) {
    return buildFilteredOptions(
      widget.editorImage,
      widget.isSelected,
      widget.filterModel.name,
      widget.onSelectFilter,
      widget.filterKey ?? ValueKey(widget.filterModel.name),
    );
  }

  Widget buildFilteredOptions(
    Widget editorImage,
    bool isSelected,
    String filterName,
    VoidCallback onSelectFilter,
    Key filterKey,
  ) {
    return GestureDetector(
      onTap: () => onSelectFilter(),
      child: SizedBox(
        height: 90,
        width: 80,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.spaceAround,
          children: [
            Container(
              height: 60,
              width: 60,
              decoration: ShapeDecoration(
                shape: SmoothRectangleBorder(
                  borderRadius: SmoothBorderRadius(
                    cornerRadius: 12,
                    cornerSmoothing: 0.6,
                  ),
                  side: BorderSide(
                    color: isSelected
                        ? Theme.of(context).colorScheme.imageEditorPrimaryColor
                        : Colors.transparent,
                    width: 1.5,
                  ),
                ),
              ),
              child: Padding(
                padding: isSelected ? const EdgeInsets.all(2) : EdgeInsets.zero,
                child: ClipSmoothRect(
                  radius: SmoothBorderRadius(
                    cornerRadius: 9.69,
                    cornerSmoothing: 0.4,
                  ),
                  child: SizedBox(
                    height: isSelected ? 56 : 60,
                    width: isSelected ? 56 : 60,
                    child: editorImage,
                  ),
                ),
              ),
            ),
            const SizedBox(height: 8),
            Text(
              filterName,
              style: isSelected
                  ? getEnteTextTheme(context).smallBold
                  : getEnteTextTheme(context).smallMuted,
              textAlign: TextAlign.center,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
            ),
          ],
        ),
      ),
    );
  }
}
